import pandas as pd
import numpy as np
import geopandas as gpd
import streamlit as st
import networkx as nx
from sqlalchemy import create_engine
from shapely import wkt
import os

def create_db_engine(db_user="postgres", db_password="postgre", db_host="localhost", db_port="5432", db_name="network_db"):
    try:
        connection_string = f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
        engine = create_engine(connection_string)
        return engine
    except Exception as e:
        st.error(f"Erreur de connexion à la base de données : {e}")
        return None

@st.cache_data
def load_traffic_data(uploaded_file, use_sample, date_start, date_end):
    if uploaded_file is not None:
        try:
            traffic_data = pd.read_csv(uploaded_file)
            st.success("Données téléchargées avec succès.")
            return traffic_data
        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier : {e}")
            return pd.DataFrame()
    elif use_sample:
        try:
            dates = pd.date_range(start=date_start, end=date_end, freq='H')
            return pd.DataFrame({
                "id_segment": np.random.choice([1, 2, 3, 4, 5], len(dates)),
                "timestamp_debut": dates,
                "volume_total": np.random.randint(50, 500, len(dates)),
                "vitesse_moy_kmh": np.random.uniform(10, 80, len(dates)),
                "ratio_saturation": np.random.uniform(0.2, 1.5, len(dates)),
                "nom_route": np.random.choice(["Route des Hydrocarbures", "Marais Masay", "Rocade d'Iarivo"], len(dates)),
                "volume_vl": np.random.randint(30, 300, len(dates)),
                "volume_pl": np.random.randint(5, 50, len(dates)),
                "volume_bus": np.random.randint(2, 20, len(dates)),
                "volume_moto": np.random.randint(10, 100, len(dates))
            })
        except Exception as e:
            st.error(f"Erreur création données échantillon : {e}")
            return pd.DataFrame()
    else:
        engine = create_db_engine(
            db_user=os.getenv("DB_USER", "postgres"),
            db_password=os.getenv("DB_PASSWORD", "postgre"),
            db_host=os.getenv("DB_HOST", "localhost"),
            db_port=os.getenv("DB_PORT", "5432"),
            db_name=os.getenv("DB_NAME", "network_db")
        )
        if engine is None:
            st.warning("Aucune connexion à la base de données. Aucune donnée fournie.")
            return pd.DataFrame()
        try:
            query = """
                SELECT id_segment, timestamp_debut, volume_total, vitesse_moy_kmh, 
                       ratio_saturation, nom_route, volume_vl, volume_pl, volume_bus, volume_moto
                FROM predictions_modele
                WHERE timestamp_debut BETWEEN %s AND %s
            """
            traffic_data = pd.read_sql(query, engine, params=(date_start, date_end))
            if traffic_data.empty:
                st.warning("Aucune donnée de trafic trouvée pour la période spécifiée.")
            else:
                st.success("Données de trafic chargées depuis la base de données.")
            return traffic_data
        except Exception as e:
            st.error(f"Erreur lors du chargement des données de trafic depuis la base de données : {e}")
            return pd.DataFrame()

def load_network_data():
    engine = create_db_engine(
        db_user=os.getenv("DB_USER", "postgres"),
        db_password=os.getenv("DB_PASSWORD", "postgre"),
        db_host=os.getenv("DB_HOST", "localhost"),
        db_port=os.getenv("DB_PORT", "5432"),
        db_name=os.getenv("DB_NAME", "network_db")
    )
    if engine is None:
        return {"G": nx.DiGraph(), "noeuds": gpd.GeoDataFrame(), "segments": pd.DataFrame()}
    
    try:
        # Load nodes data as GeoDataFrame to handle PostGIS geometry
        nodes_query = "SELECT id_noeud, nom, ST_Transform(geometrie, 4326) AS geometry , flux_entrant, flux_sortant, nb_voies, nb_intersections, nb_embranchements, capacite FROM noeuds"
        nodes_gdf = gpd.read_postgis(nodes_query, engine, geom_col="geometry")
        
        # Validate geometry types
        if not nodes_gdf.empty:
            geometry_types = nodes_gdf.geometry.geom_type.unique()
            if not all(geom_type == 'Point' for geom_type in geometry_types):
                st.warning(f"Types de géométrie inattendus dans la table noeuds : {geometry_types}. Seuls POINT sont attendus.")
        
        # Ensure CRS is EPSG:4326
        nodes_gdf.set_crs(epsg=4326, inplace=True, allow_override=True)
        
        # Load segments data with geometry and additional columns
        segments_query = """
            SELECT id_segment, id_noeud_amont, id_noeud_aval, longueur_metres, nom_route, 
                   sens_unique, vitesse_ref_kmh, type_segment, nb_voies, 
                   ST_AsText(ST_Transform(geometrie, 4326)) AS geometrie       FROM segments
        """
        segments_df = pd.read_sql(segments_query, engine)
        
        # Convert WKT geometries to Shapely objects
        if 'geometrie' in segments_df.columns:
            def parse_wkt(geom_str):
                try:
                    if pd.isna(geom_str) or not geom_str:
                        return None
                    return wkt.loads(geom_str)
                except Exception as e:
                    st.warning(f"Erreur lors du décodage de la géométrie WKT pour un segment : {e}")
                    return None
            
            segments_df['geometrie'] = segments_df['geometrie'].apply(parse_wkt)
            
            # Validate geometry types
            valid_geometries = segments_df['geometrie'].notnull()
            if valid_geometries.sum() == 0:
                st.warning("Aucune géométrie valide dans la table segments.")
            else:
                geometry_types = segments_df.loc[valid_geometries, 'geometrie'].apply(lambda x: x.geom_type).unique()
                if not all(geom_type == 'LineString' for geom_type in geometry_types):
                    st.warning(f"Types de géométrie inattendus dans la table segments : {geometry_types}. Seuls LINESTRING sont attendus.")
        
        # Create network graph
        G = nx.DiGraph()
        for _, row in nodes_gdf.iterrows():
            lon, lat = row["geometry"].x, row["geometry"].y
            G.add_node(row["id_noeud"], name=row["nom"], pos=(lat, lon))
        
        for _, row in segments_df.iterrows():
            # Add edge only if segment has valid geometry or required attributes
            if pd.notna(row["id_noeud_amont"]) and pd.notna(row["id_noeud_aval"]):
                G.add_edge(
                    row["id_noeud_amont"], 
                    row["id_noeud_aval"], 
                    id_segment=row["id_segment"], 
                    length=row["longueur_metres"],
                    nom_route=row["nom_route"],
                    sens_unique=row.get("sens_unique", False),
                    vitesse_ref_kmh=row.get("vitesse_ref_kmh", None),
                    type_segment=row.get("type_segment", None),
                    nb_voies=row.get("nb_voies", None)
                )
        
        return {
            "G": G,
            "noeuds": nodes_gdf,
            "segments": segments_df
        }
    except Exception as e:
        st.error(f"Erreur lors du chargement des données réseau : {e}")
        return {"G": nx.DiGraph(), "noeuds": gpd.GeoDataFrame(), "segments": pd.DataFrame()}


def load_flux_data():
    engine = create_db_engine(
        db_user=os.getenv("DB_USER", "postgres"),
        db_password=os.getenv("DB_PASSWORD", "postgre"),
        db_host=os.getenv("DB_HOST", "localhost"),
        db_port=os.getenv("DB_PORT", "5432"),
        db_name=os.getenv("DB_NAME", "network_db")
    )
    if engine is None:
        return pd.DataFrame()
    
    try:
        flux_query = "SELECT * FROM od_Ankorondrano"
        df_flux = pd.read_sql(flux_query, engine)
        return df_flux
    except Exception as e:
        st.error(f"Erreur lors du chargement des données de flux : {e}")
        return pd.DataFrame()