import networkx as nx
import pandas as pd
import numpy as np
import streamlit as st
import geopandas as gpd
from shapely.geometry import Point, LineString
import folium
from folium.plugins import MarkerCluster, PolyLineTextPath
from shapely import wkt
import matplotlib.pyplot as plt
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
from utils.data_loader import load_network_data, load_flux_data

def get_legend_html():
    return '''
        <div style="position: absolute; 
                    bottom: 10px; left: 10px; 
                    background-color: white; 
                    border: 2px solid grey; 
                    z-index: 9999; 
                    font-size: 14px; 
                    padding: 10px; 
                    border-radius: 5px;
                    max-height: 200px; 
                    overflow-y: auto;">
            <p><strong>Légende du trafic</strong></p>
            <div style="display: flex; align-items: center;">
                <div style="background: blue; width: 20px; height: 20px; border-radius: 50%; margin-right: 10px;"></div>
                <span>Fluide (ratio < 0.6)</span>
            </div>
            <div style="display: flex; align-items: center;">
                <div style="background: orange; width: 20px; height: 20px; border-radius: 50%; margin-right: 10px;"></div>
                <span>Modéré (ratio 0.6-0.8)</span>
            </div>
            <div style="display: flex; align-items: center;">
                <div style="background: red; width: 20px; height: 20px; border-radius: 50%; margin-right: 10px;"></div>
                <span>Saturé (ratio > 0.8)</span>
            </div>
            <div style="display: flex; align-items: center;">
                <div style="background: gray; width: 20px; height: 2px; margin-right: 10px;"></div>
                <span>Segment de route (estompé avec flèches)</span>
            </div>
        </div>
    '''

def optimize_route(G, start_node, end_node, predictions, segments):
    for u, v, d in G.edges(data=True):
        segment_id = d["id_segment"]
        saturation = np.random.uniform(0.5, 1.5)
        d["weight"] = d["length"] * (1 + saturation)
    try:
        return nx.shortest_path(G, start_node, end_node, weight="weight")
    except nx.NetworkXNoPath:
        return None

def get_path_segments(path, segments):
    path_segments = []
    for i in range(len(path) - 1):
        start = path[i]
        end = path[i + 1]
        segment = segments[
            ((segments["id_noeud_amont"] == start) & (segments["id_noeud_aval"] == end)) |
            ((segments["id_noeud_amont"] == end) & (segments["id_noeud_aval"] == start) & (segments["sens_unique"] == False))
        ]
        if not segment.empty:
            path_segments.append(segment.iloc[0])
    return pd.DataFrame(path_segments)

def show_osm_nodes():
    try:
        st.markdown(
            """
            <style>
            iframe[src*="leaflet"] {
                height: 100% !important;
                min-height: 600px;
                margin: 0 !important;
                padding: 0 !important;
            }
            .stApp {
                margin: 0;
                padding: 0;
            }
            .css-1aumxhk {
                margin-bottom: 0 !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        network_data = load_network_data()
        nodes_gdf = network_data["noeuds"]
        segments_df = network_data["segments"]
        G = network_data["G"]
        
        if nodes_gdf.empty:
            st.error("Aucune donnée de nœuds disponible.")
            return

        if segments_df.empty:
            st.warning("Aucune donnée de segments disponible. Affichage sans segments.")
            return
        
        if 'geometrie' in segments_df.columns:
            if segments_df['geometrie'].dtype == 'object' and segments_df['geometrie'].apply(lambda x: isinstance(x, str) and x.startswith('LINESTRING')).any():
                st.info("Détection de géométries WKT. Conversion en objets Shapely...")
                def parse_wkt(geom_str):
                    try:
                        if pd.isna(geom_str) or not geom_str:
                            return None
                        return wkt.loads(geom_str)
                    except Exception as e:
                        st.warning(f"Erreur lors du décodage de la géométrie WKT : {e}")
                        return None
                segments_df['geometrie'] = segments_df['geometrie'].apply(parse_wkt)
            elif segments_df['geometrie'].isnull().all():
                st.warning("La colonne 'geometrie' contient uniquement des valeurs nulles.")
                if 'geometrie' in segments_df.columns:
                    segments_df = segments_df.drop(columns=['geometrie'])
        else:
            st.warning("La colonne 'geometrie' est absente.")
            segments_df = pd.DataFrame()

        if not segments_df.empty and 'geometrie' in segments_df.columns:
            valid_geoms = segments_df['geometrie'].notnull().sum()
            if valid_geoms == 0:
                st.warning("Aucune géométrie valide après conversion.")
                segments_df = pd.DataFrame()
            else:
                st.info(f"{valid_geoms} segments avec géométries valides détectés.")
        else:
            st.warning("Aucun segment à afficher.")

        if not all(col in nodes_gdf.columns for col in ['flux_entrant', 'flux_sortant', 'nb_voies', 'nb_intersections', 'nb_embranchements', 'capacite']):
            st.error("Les colonnes de trafic (flux_entrant, flux_sortant, nb_voies, nb_intersections, nb_embranchements, capacite) sont manquantes dans la table noeuds.")
            return

        nodes_gdf["ratio_saturation"] = nodes_gdf["flux_sortant"] / (nodes_gdf["capacite"] + 1)
        
        def get_service_level(ratio):
            if ratio < 0.3: return "A"
            elif ratio < 0.5: return "B"
            elif ratio < 0.7: return "C"
            elif ratio < 0.8: return "D"
            elif ratio < 0.9: return "E"
            else: return "F"
        nodes_gdf["niveau_service"] = nodes_gdf["ratio_saturation"].apply(get_service_level)
        
        def get_color(ratio):
            if ratio < 0.6: return "blue"
            elif ratio < 0.8: return "orange"
            else: return "red"
        nodes_gdf["couleur"] = nodes_gdf["ratio_saturation"].apply(get_color)

        nodes_gdf.set_crs(epsg=4326, inplace=True, allow_override=True)

        center_lat = nodes_gdf.geometry.y.mean()
        center_lon = nodes_gdf.geometry.x.mean()
        
        st.markdown("---")
        col_center1, col_center2 = st.columns(2)
        with col_center1:
            st.metric("Centre Latitude", f"{center_lat:.6f}° S" if center_lat < 0 else f"{center_lat:.6f}° N", delta=None)
        with col_center2:
            st.metric("Centre Longitude", f"{center_lon:.6f}° E" if center_lon > 0 else f"{center_lon:.6f}° W", delta=None)
        st.markdown("---")

        m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="CartoDB positron", control_scale=True)
        m.get_root().html.add_child(folium.Element(get_legend_html()))

        if not segments_df.empty and 'geometrie' in segments_df.columns:
            valid_geoms = 0
            for idx, row in segments_df.iterrows():
                try:
                    if row['geometrie'] is None or not hasattr(row['geometrie'], 'coords'):
                        st.warning(f"Segment {row['id_segment']} ignoré : géométrie invalide.")
                        continue
                    geom = row['geometrie']
                    coords = list(geom.coords)
                    coords_folium = [(lat, lon) for lon, lat in coords]
                    segment_line = folium.PolyLine(
                        locations=coords_folium,
                        color="gray",
                        weight=10,
                        opacity=0.3,
                        popup=f"Segment {row['id_segment']}: {row['nom_route']} ({row['longueur_metres']}m)"
                    )
                    segment_line.add_to(m)
                    if row.get('sens_unique', True):
                        PolyLineTextPath(
                            segment_line,
                            "→",
                            repeat=True,
                            offset=10,
                            attributes={"fill": "gray", "font-size": "16"}
                        ).add_to(m)
                    else:
                        PolyLineTextPath(
                            segment_line,
                            "→",
                            repeat=True,
                            offset=10,
                            attributes={"fill": "gray", "font-size": "16"}
                        ).add_to(m)
                        PolyLineTextPath(
                            segment_line,
                            "←",
                            repeat=True,
                            offset=-10,
                            attributes={"fill": "gray", "font-size": "16"}
                        ).add_to(m)
                    valid_geoms += 1
                except Exception as seg_error:
                    st.warning(f"Erreur lors de l'ajout du segment {row['id_segment']} ({row['nom_route']}): {seg_error}")
            st.info(f"{valid_geoms} segments valides affichés sur la carte.")
        else:
            st.warning("Aucun segment à afficher.")

        for idx, row in nodes_gdf.iterrows():
            marker_id = f"node_{row['id_noeud']}"
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=10 + row['ratio_saturation'] * 5,
                tooltip=f"Cliquez pour les détails: {row['nom']}",
                popup=folium.Popup(f"ID: {row['id_noeud']}", max_width=100),
                color=row['couleur'],
                fill=True,
                fill_color=row['couleur'],
                fill_opacity=0.7,
                weight=2
            ).add_to(m)

        col1, col2 = st.columns([2, 1])
        with col1:
            map_data = st_folium(m, width=700, height=600, key="osm_map")
        
        with col2:
            st.subheader("📋 Détails du Nœud")
            def find_closest_node(click_lat, click_lon):
                min_distance = float('inf')
                closest_node = None
                for _, row in nodes_gdf.iterrows():
                    distance = math.sqrt((row.geometry.y - click_lat)**2 + (row.geometry.x - click_lon)**2)
                    if distance < min_distance:
                        min_distance = distance
                        closest_node = row
                if min_distance < 0.02:
                    return closest_node
                return None
            
            if 'selected_node' not in st.session_state:
                st.session_state.selected_node = None
            
            if map_data and 'last_clicked' in map_data and map_data['last_clicked']:
                clicked_lat = map_data['last_clicked'].get('lat')
                clicked_lon = map_data['last_clicked'].get('lng')
                if clicked_lat is not None and clicked_lon is not None:
                    closest_node = find_closest_node(clicked_lat, clicked_lon)
                    if closest_node is not None:
                        st.session_state.selected_node = closest_node
            
            if st.session_state.selected_node is not None:
                node = st.session_state.selected_node
                st.markdown(f"### {node['nom']}")
                st.markdown(f"**ID:** {node['id_noeud']}")
                st.markdown("---")
                st.markdown("#### 🏗️ Caractéristiques")
                st.markdown(f"- **Nombre de voies:** {node['nb_voies']}")
                st.markdown(f"- **Intersections:** {node['nb_intersections']}")
                st.markdown(f"- **Embranchements:** {node['nb_embranchements']}")
                st.markdown(f"- **Capacité:** {node['capacite']} véh/h")
                st.markdown("---")
                st.markdown("#### 🚦 Trafic")
                st.markdown(f"- **Flux entrants:** {node['flux_entrant']} véh/h")
                st.markdown(f"- **Flux sortants:** {node['flux_sortant']} véh/h")
                st.markdown(f"- **Ratio V/C:** {node['ratio_saturation']:.2f}")
                st.markdown(f"- **Niveau de service:** {node['niveau_service']}")
                st.markdown("---")
                st.markdown("#### 💡 Insights")
                if node['ratio_saturation'] < 0.6:
                    st.success("✅ Trafic fluide - Bonnes conditions")
                elif node['ratio_saturation'] < 0.8:
                    st.warning("⚠️ Trafic modéré - Attention aux heures de pointe")
                else:
                    st.error("❌ Trafic saturé - Nécessite des aménagements")
                if node['niveau_service'] in ['A', 'B']:
                    st.info("📊 Faible congestion - Circulation optimale")
                elif node['niveau_service'] in ['C', 'D']:
                    st.info("📊 Congestion modérée - Ralentissements possibles")
                else:
                    st.info("📊 Forte congestion - Bouchons fréquents")
                if st.button("Effacer la sélection"):
                    st.session_state.selected_node = None
                    st.rerun()
            else:
                st.info("👆 Cliquez sur un nœud de la carte pour afficher ses détails")
            
            st.markdown("---")
            st.markdown("#### ℹ️ Légende")
            st.markdown("""
            **Niveaux de service:**
            - **A**: Circulation libre
            - **B**: Circulation fluide
            - **C**: Circulation stable
            - **D**: Congestion fréquente
            - **E**: Saturation
            - **F**: Congestion maximale
            
            **Couleurs:**
            - 🔵 Fluide (ratio < 0.6)
            - 🟠 Modéré (ratio 0.6-0.8)
            - 🔴 Saturé (ratio > 0.8)
            - 🟨 Segment de route (estompé avec flèches)
            """)

        with st.expander("📊 Tableau Analytique des Nœuds"):
            col1, col2 = st.columns([2, 1])
            with col1:
                debug_data = nodes_gdf[['id_noeud', 'nom', 'flux_entrant', 'flux_sortant', 'ratio_saturation']].copy()
                debug_data['latitude'] = nodes_gdf.geometry.y
                debug_data['longitude'] = nodes_gdf.geometry.x
                debug_data['average_traffic'] = (debug_data['flux_entrant'] + debug_data['flux_sortant']) / 2
                debug_data['congestion_score'] = debug_data['ratio_saturation'] * 100
                
                def traffic_color(val):
                    if val < 1000:
                        r = int(144 + (0 / 1000) * (255 - 144))
                        g = int(238 + (255 / 1000) * (0 - 238))
                        b = int(144 + (0 / 1000) * (0 - 144))
                    elif val < 2000:
                        r = int(255 + (0 / 1000) * (255 - 255))
                        g = int(165 + (165 / 1000) * (0 - 165))
                        b = int(0 + (0 / 1000) * (0 - 0))
                    else:
                        r = int(255 + (0 / 1000) * (0 - 255))
                        g = int(0 + (0 / 1000) * (0 - 0))
                        b = int(0 + (0 / 1000) * (0 - 0))
                    return f'background-color: rgb({r}, {g}, {b})'
                
                def congestion_color(val):
                    if val < 30:
                        r = int(173 + (0 / 70) * (138 - 173))
                        g = int(216 + (0 / 70) * (43 - 216))
                        b = int(230 + (0 / 70) * (226 - 230))
                    elif val < 70:
                        r = int(138 + (0 / 40) * (148 - 138))
                        g = int(43 + (0 / 40) * (0 - 43))
                        b = int(226 + (0 / 40) * (211 - 226))
                    else:
                        r = int(148 + (0 / 30) * (0 - 148))
                        g = int(0 + (0 / 30) * (0 - 0))
                        b = int(211 + (0 / 30) * (0 - 211))
                    return f'background-color: rgb({r}, {g}, {b})'
                
                styled_data = debug_data.style.applymap(traffic_color, subset=['average_traffic'])
                styled_data = styled_data.applymap(congestion_color, subset=['congestion_score'])
                st.dataframe(styled_data)
            with col2:
                st.markdown("**Notes et unités :**")
                st.markdown("- **🚗 flux_entrant / flux_sortant** : véh/h.")
                st.markdown("- **⚖️ ratio_saturation** : 0 (libre) → 1 (saturé).")
                st.markdown("- *niveau_service* : Lettres indiquent les niveaux de service :")
                st.markdown("  - A : Circulation libre /  - B : Circulation fluide")
                st.markdown("  - C : Circulation stable  / - D : Congestion fréquente")
                st.markdown("  - E : Saturation  /  - F : Congestion maximale")
                st.markdown("- **average_traffic** : 🟢 (faible < 1000 véh/j) → 🟠 → 🔴 (élevé > 2000 véh/j).")
                st.markdown("- **congestion_score** : Dégradé de bleu clair (faible congestion < 30%) à violet foncé (forte congestion > 70%).")

        with st.expander("🛣️ Itinéraires Possibles"):
            has_valid_segments = not segments_df.empty and 'geometrie' in segments_df.columns and segments_df['geometrie'].notnull().sum() > 0
            if not has_valid_segments:
                st.warning("Impossible de calculer des itinéraires sans segments chargés ou géométries valides.")
            else:
                st.markdown("Sélectionnez un départ et une arrivée pour visualiser un itinéraire possible et ses segments.")
                col_start, col_end = st.columns(2)
                with col_start:
                    start_node = st.selectbox("Nœud de départ", options=nodes_gdf['nom'].tolist(), key="start_node")
                    start_id = nodes_gdf[nodes_gdf['nom'] == start_node]['id_noeud'].iloc[0]
                with col_end:
                    end_node = st.selectbox("Nœud d'arrivée", options=nodes_gdf['nom'].tolist(), key="end_node")
                    end_id = nodes_gdf[nodes_gdf['nom'] == end_node]['id_noeud'].iloc[0]
                
                if 'route_path' not in st.session_state:
                    st.session_state.route_path = None
                if 'route_segments' not in st.session_state:
                    st.session_state.route_segments = None
                if 'route_map' not in st.session_state:
                    st.session_state.route_map = None
                
                if st.button("Calculer Itinéraire"):
                    if start_id == end_id:
                        st.warning("Veuillez sélectionner des nœuds différents.")
                    else:
                        with st.spinner("Calcul de l'itinéraire..."):
                            path = optimize_route(G, start_id, end_id, None, segments_df)
                            if path is not None:
                                path_segments = get_path_segments(path, segments_df)
                                if not path_segments.empty:
                                    st.session_state.route_path = path
                                    st.session_state.route_segments = path_segments
                                    total_length = path_segments['longueur_metres'].sum()
                                    
                                    m_path = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="CartoDB positron", control_scale=True)
                                    
                                    for idx, row in segments_df.iterrows():
                                        try:
                                            if pd.isna(row['geometrie']) or not hasattr(row['geometrie'], 'coords'):
                                                continue
                                            geom = row['geometrie']
                                            coords = list(geom.coords)
                                            coords_folium = [(lat, lon) for lon, lat in coords]
                                            segment_line = folium.PolyLine(
                                                locations=coords_folium,
                                                color="gray",
                                                weight=10,
                                                opacity=0.3,
                                                popup=f"Segment {row['id_segment']}: {row['nom_route']}"
                                            )
                                            segment_line.add_to(m_path)
                                            if row.get('sens_unique', True):
                                                PolyLineTextPath(
                                                    segment_line,
                                                    "→",
                                                    repeat=True,
                                                    offset=10,
                                                    attributes={"fill": "gray", "font-size": "16"}
                                                ).add_to(m_path)
                                            else:
                                                PolyLineTextPath(
                                                    segment_line,
                                                    "→",
                                                    repeat=True,
                                                    offset=10,
                                                    attributes={"fill": "gray", "font-size": "16"}
                                                ).add_to(m_path)
                                                PolyLineTextPath(
                                                    segment_line,
                                                    "←",
                                                    repeat=True,
                                                    offset=-10,
                                                    attributes={"fill": "gray", "font-size": "16"}
                                                ).add_to(m_path)
                                        except Exception as seg_error:
                                            st.warning(f"Erreur lors de l'ajout du segment de fond {row['id_segment']} : {seg_error}")
                                    
                                    for idx, row in nodes_gdf.iterrows():
                                        folium.CircleMarker(
                                            location=[row.geometry.y, row.geometry.x],
                                            radius=10 + row['ratio_saturation'] * 5,
                                            tooltip=f"Cliquez pour les détails: {row['nom']}",
                                            popup=folium.Popup(f"ID: {row['id_noeud']}", max_width=100),
                                            color=row['couleur'],
                                            fill=True,
                                            fill_color=row['couleur'],
                                            fill_opacity=0.7,
                                            weight=2
                                        ).add_to(m_path)
                                    
                                    for idx, row in path_segments.iterrows():
                                        try:
                                            if pd.isna(row['geometrie']) or not hasattr(row['geometrie'], 'coords'):
                                                st.warning(f"Segment d'itinéraire {row['id_segment']} ignoré : géométrie invalide.")
                                                continue
                                            geom = row['geometrie']
                                            coords = list(geom.coords)
                                            coords_folium = [(lat, lon) for lon, lat in coords]
                                            path_line = folium.PolyLine(
                                                locations=coords_folium,
                                                color="blue",
                                                weight=4,
                                                opacity=0.8,
                                                popup=f"Segment {row['id_segment']}: {row['nom_route']}"
                                            )
                                            path_line.add_to(m_path)
                                            if row.get('sens_unique', True):
                                                PolyLineTextPath(
                                                    path_line,
                                                    "→",
                                                    repeat=True,
                                                    offset=10,
                                                    attributes={"fill": "blue", "font-size": "16"}
                                                ).add_to(m_path)
                                            else:
                                                PolyLineTextPath(
                                                    path_line,
                                                    "→",
                                                    repeat=True,
                                                    offset=10,
                                                    attributes={"fill": "blue", "font-size": "16"}
                                                ).add_to(m_path)
                                                PolyLineTextPath(
                                                    path_line,
                                                    "←",
                                                    repeat=True,
                                                    offset=-10,
                                                    attributes={"fill": "blue", "font-size": "16"}
                                                ).add_to(m_path)
                                        except Exception as path_error:
                                            st.warning(f"Erreur lors de l'ajout du segment d'itinéraire {row['id_segment']} : {path_error}")
                                    
                                    m_path.get_root().html.add_child(folium.Element(get_legend_html()))
                                    st.session_state.route_map = m_path
                                    st.success(f"Itinéraire trouvé : {' → '.join([str(n) for n in path])}")
                                    display_cols = ['id_segment', 'nom_route', 'longueur_metres'] + \
                                                   [col for col in ['vitesse_ref_kmh', 'type_segment', 'nb_voies'] if col in path_segments.columns]
                                    st.table(path_segments[display_cols])
                                    st.info(f"**Longueur totale estimée : {total_length} mètres**")
                                else:
                                    st.warning("Aucun segment trouvé pour cet itinéraire.")
                                    st.session_state.route_path = None
                                    st.session_state.route_segments = None
                                    st.session_state.route_map = None
                            else:
                                st.error("Aucun itinéraire trouvé entre ces nœuds.")
                                st.session_state.route_path = None
                                st.session_state.route_segments = None
                                st.session_state.route_map = None
                
                if st.session_state.route_path and st.session_state.route_segments is not None and st.session_state.route_map:
                    st.success(f"Itinéraire persistant : {' → '.join([str(n) for n in st.session_state.route_path])}")
                    display_cols = ['id_segment', 'nom_route', 'longueur_metres'] + \
                                   [col for col in ['vitesse_ref_kmh', 'type_segment', 'nb_voies'] if col in st.session_state.route_segments.columns]
                    st.table(st.session_state.route_segments[display_cols])
                    total_length = st.session_state.route_segments['longueur_metres'].sum()
                    st.info(f"**Longueur totale estimée : {total_length} mètres**")
                    st_folium(st.session_state.route_map, width=700, height=600, key="persisted_path_map")
                
                if st.button("Effacer Itinéraire"):
                    st.session_state.route_path = None
                    st.session_state.route_segments = None
                    st.session_state.route_map = None
                    st.rerun()

    except Exception as e:
        st.error(f"Erreur lors de l'affichage de la carte : {e}")

def show_flux_graph_from_ankorondrano():
    try:
        st.markdown("#### 📊 Flux d'échanges entre Ankorondrano et les autres arrondissements")
        
        df_flux = load_flux_data()
        
        if df_flux.empty:
            st.error("Aucune donnée de flux disponible.")
            return
        
        df_flux.columns = df_flux.columns.str.strip()
        df_flux = df_flux[df_flux["arrondissement"] != "1er"]
        
        fig_flux = make_subplots(rows=1, cols=2, 
                                 subplot_titles=("Flux entrants vers Ankorondrano", "Flux sortants depuis Ankorondrano"),
                                 horizontal_spacing=0.2)
        
        fig_flux.add_trace(
            go.Bar(x=df_flux["arrondissement"], 
                   y=df_flux["Flux depuis 1er"], 
                   name="Entrants",
                   marker_color="lightblue",
                   hovertemplate="<b>%{x}</b><br>Flux entrant: %{y:,} véh/j<extra></extra>"),
            row=1, col=1
        )
        
        fig_flux.add_trace(
            go.Bar(x=df_flux["arrondissement"], 
                   y=df_flux["Flux vers 1er"], 
                   name="Sortants",
                   marker_color="lightcoral",
                   hovertemplate="<b>%{x}</b><br>Flux sortant: %{y:,} véh/j<extra></extra>"),
            row=1, col=2
        )
        
        fig_flux.update_layout(
            height=400,
            showlegend=False,
            title_text="Flux journaliers entre Ankorondrano et les autres arrondissements",
            title_x=0.5
        )
        
        fig_flux.update_yaxes(title_text="Véhicules/jour", row=1, col=1)
        fig_flux.update_yaxes(title_text="Véhicules/jour", row=1, col=2)
        
        st.plotly_chart(fig_flux, use_container_width=True)
        
        fig_balance = go.Figure()
        colors = ["green" if val >= 0 else "red" for val in df_flux["Balance (Sortants – Entrants)"]]
        
        fig_balance.add_trace(
            go.Bar(x=df_flux["arrondissement"], 
                   y=df_flux["Balance (Sortants – Entrants)"], 
                   name="Balance",
                   marker_color=colors,
                   text=df_flux["Balance (Sortants – Entrants)"],
                   texttemplate="%{text:,}",
                   textposition="outside",
                   hovertemplate="<b>%{x}</b><br>Balance: %{y:,} véh/j<extra></extra>")
        )
        
        fig_balance.update_layout(
            title="Balance des flux (Sortants - Entrants) par arrondissement",
            xaxis_title="arrondissement",
            yaxis_title="Balance (véhicules/jour)",
            height=400,
            title_x=0.5
        )
        
        fig_balance.add_hline(y=0, line_width=2, line_dash="dash", line_color="gray")
        
        st.plotly_chart(fig_balance, use_container_width=True)
        
        G = nx.DiGraph()
        for _, row in df_flux.iterrows():
            arrondissement = row["arrondissement"]
            flux_entrant = row["Flux depuis 1er"]
            flux_sortant = row["Flux vers 1er"]
            balance = row["Balance (Sortants – Entrants)"]
            G.add_node(arrondissement)
            if flux_entrant > 0:
                G.add_edge(arrondissement, "Ankorondrano", weight=flux_entrant, type="entrant")
            if flux_sortant > 0:
                G.add_edge("Ankorondrano", arrondissement, weight=flux_sortant, type="sortant")
        
        pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
        pos["Ankorondrano"] = [0, 0]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=2000, ax=ax)
        nx.draw_networkx_labels(G, pos, ax=ax)
        
        edge_colors = []
        edge_widths = []
        for u, v, data in G.edges(data=True):
            if data['type'] == 'entrant':
                edge_colors.append('red')
            else:
                edge_colors.append('green')
            edge_widths.append(data['weight'] / 5000)
        
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, 
                              arrows=True, arrowsize=20, ax=ax)
        
        edge_labels = {(u, v): f"{d['weight']:,.0f}" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax)
        
        ax.set_title("Flux entre Ankorondrano et les autres arrondissements")
        ax.axis('off')
        
        st.pyplot(fig)
        
        st.markdown("---")
        st.markdown("#### 📈 Analyse des flux d'Ankorondrano")
        
        total_entrants = df_flux["Flux depuis 1er"].sum()
        total_sortants = df_flux["Flux vers 1er"].sum()
        balance_totale = total_sortants - total_entrants
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Flux entrants totaux", f"{total_entrants:,.0f} véh/j")
        with col2:
            st.metric("Flux sortants totaux", f"{total_sortants:,.0f} véh/j")
        with col3:
            st.metric("Balance totale", f"{balance_totale:,.0f} véh/j", 
                     delta_color="inverse" if balance_totale < 0 else "normal")
        
        st.markdown("##### 💡 Insights principaux")
        max_entrant = df_flux.loc[df_flux["Flux depuis 1er"].idxmax()]
        max_sortant = df_flux.loc[df_flux["Flux vers 1er"].idxmax()]
        max_balance = df_flux.loc[df_flux["Balance (Sortants – Entrants)"].idxmax()]
        origine_text = "d'origine" if balance_totale > 0 else "de destination"
        trafic_text = "plus de départs que d'arrivées" if balance_totale > 0 else "plus d'arrivées que de départs"
        
        st.info(f"""
        **Caractéristiques des flux d'Ankorondrano:**
        
        - 🏙️ **Destination principale**: Le {max_sortant['arrondissement']} arrondissement est la destination la plus fréquente depuis Ankorondrano avec {max_sortant['Flux vers 1er']:,.0f} véhicules/jour.
        
        - 🚗 **Origine principale**: Le {max_entrant['arrondissement']} arrondissement est la principale source de trafic vers Ankorondrano avec {max_entrant['Flux depuis 1er']:,.0f} véhicules/jour.
        
        - ⚖️ **Plus grand déséquilibre**: Le {max_balance['arrondissement']} arrondissement présente le plus grand écart de flux ({max_balance['Balance (Sortants – Entrants)']:,.0f} véh/j), indiquant une forte attractivité ou un important trafic de transit.
        
        - 📊 **Profil global**: Ankorondrano est principalement une zone {origine_text} du trafic, 
        avec {trafic_text}.
        """)
        
        st.markdown("##### 🎯 Recommandations")
        if balance_totale > 0:
            st.success("""
            **Ankorondrano comme zone d'origine du trafic:**
            - Développer des infrastructures pour fluidifier les départs aux heures de pointe
            - Optimiser les axes de sortie de la zone
            - Envisager des solutions de transport en commun pour réduire la congestion aux sorties
            """)
        else:
            st.success("""
            **Ankorondrano comme zone de destination du trafic:**
            - Améliorer les capacités de stationnement
            - Optimiser les axes d'entrée dans la zone
            - Développer des solutions de mobilité douce pour les derniers kilomètres
            """)
            
    except Exception as e:
        st.error(f"Erreur lors de l'affichage des flux : {e}")
        st.info("Assurez-vous que la table 'od_Ankorondrano' est présente dans la base de données avec le bon format.")