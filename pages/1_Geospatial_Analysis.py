import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
import matplotlib.pyplot as plt
import altair as alt
import plotly.express as px
from datetime import datetime, timedelta

from style import apply_custom_css, load_google_fonts
from utils.data_loader import load_traffic_data, load_network_data
from utils.routing import optimize_route
from utils.routing import show_osm_nodes, show_flux_graph_from_ankorondrano

# Configuration de la page
st.set_page_config(page_title="Optimisation d'Itinéraires - Antananarivo", layout="wide")

# Fonts et styles
load_google_fonts()
apply_custom_css()

# Titre principal
st.markdown("""
<div style="background: linear-gradient(135deg, #1f77b4, #4CAF50); padding: 20px; border-radius: 10px;">
    <h1 style="color: white; text-align: center;">Modélisation et Prédiction des Choix d'Itinéraires</h1>
    <h2 style="color: white; text-align: center;">Optimisation des Flux de Trafic à Antananarivo</h2>
</div>
""", unsafe_allow_html=True)


# Sidebar: Importation des données
st.sidebar.header("Importation des Données")
uploaded_file = st.sidebar.file_uploader("Télécharger données trafic (CSV)", type=["csv"])
use_sample = st.sidebar.checkbox("Utiliser données échantillon", value=True)

# Sidebar: Plage de dates
st.sidebar.header("Plage de Dates")
date_start = st.sidebar.date_input("Date début", value=datetime.now() - timedelta(days=7))
date_end = st.sidebar.date_input("Date fin", value=datetime.now())



# Analyse géospatiale et flux
st.markdown("### 🌍 Analyse Géospatiale & Flux d'Échanges")
tab1, tab2 = st.tabs(["🌍 Analyse Géospatiale", "📊 Flux d'Échanges"])
with tab1:
    show_osm_nodes()
with tab2:
    show_flux_graph_from_ankorondrano()

# Chargement des données
with st.spinner("Chargement des données de trafic..."):
    traffic_data = load_traffic_data(uploaded_file, use_sample, date_start, date_end)

if traffic_data.empty:
    st.warning("Aucune donnée disponible.")
    st.stop()

# Filtrage
traffic_data["timestamp_debut"] = pd.to_datetime(traffic_data["timestamp_debut"], errors='coerce')
filtered_data = traffic_data[
    (traffic_data["timestamp_debut"].dt.date >= date_start) &
    (traffic_data["timestamp_debut"].dt.date <= date_end)
]

# Aperçu
with st.expander("📊 Données consolidées des segments et leur signification"):
    # Création de deux colonnes pour le tableau et la légende
    col1, col2 = st.columns([3, 2])  # Ratio 3:2 pour équilibrer l'affichage

    # Affichage du DataFrame dans la première colonne
    with col1:
        st.dataframe(filtered_data.head())

    # Légende améliorée avec HTML et CSS dans la deuxième colonne
    with col2:
        legend_html = '''
        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; border: 1px solid #ddd; height: 100%;">
            <p><strong>📜 Signification des colonnes</strong></p>
            <p><strong>id_segment</strong>: Identifiant unique du segment de route.</p>
            <p><strong>timestamp_debut</strong>: Début de la période d'observation (AAAA-MM-JJ HH:MM:SS).</p>
            <p><strong>volume_total</strong>: Nombre total de véhicules sur le segment durant la période. 🚗</p>
            <p><strong>vitesse_moy_kmh</strong>: Vitesse moyenne des véhicules (km/h). 💨</p>
            <p><strong>ratio_saturation</strong>: Niveau de congestion (entre 0 et 1). Proche de 1 = congestion élevée. 🚦</p>
            <p><strong>nom_route</strong>: Nom de la route ou de la section concernée. 🛣️</p>
        </div>
        '''
        st.markdown(legend_html, unsafe_allow_html=True)

# Métriques
st.markdown("### 📊 Aperçu des Métriques Clés")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Volume Total", f"{filtered_data['volume_total'].sum():.0f}", "+5%")
with col2:
    st.metric("Vitesse Moyenne", f"{filtered_data['vitesse_moy_kmh'].mean():.2f}", "-2%")
with col3:
    st.metric("Ratio Saturation", f"{filtered_data['ratio_saturation'].mean():.2f}", "+10%")

# Visualisations
st.markdown("### 📈 Visualisations")
col1, col2 = st.columns(2)
with col1:
    freq = st.selectbox("Fréquence", ["Daily", "Weekly", "Monthly"])
    resample_rule = {'Daily': 'D', 'Weekly': 'W', 'Monthly': 'M'}[freq]
    time_series = filtered_data.set_index("timestamp_debut")["volume_total"].resample(resample_rule).sum()
    st.plotly_chart(px.line(time_series, title="Volume de Trafic Temporel"), use_container_width=True)

with col2:
    vehicle_data = {
        "VL": filtered_data.get("volume_vl", pd.Series()).sum(),
        "PL": filtered_data.get("volume_pl", pd.Series()).sum(),
        "Bus": filtered_data.get("volume_bus", pd.Series()).sum(),
        "Moto": filtered_data.get("volume_moto", pd.Series()).sum(),
    }
    st.plotly_chart(px.pie(names=list(vehicle_data.keys()), values=list(vehicle_data.values()), title="Types de Véhicules"), use_container_width=True)

# Paramètres de trajet
st.markdown("### 📍 Saisie des Paramètres de Trajet")
col1, col2 = st.columns(2)
with col1:
    start = st.selectbox("Départ", ["Ivandry La City", "Ankorondrano", "Nanisana", "Tsarasaotra"])
with col2:
    end = st.selectbox("Arrivée", ["Ivandry La City", "Ankorondrano", "Nanisana", "Tsarasaotra"])
hour = st.time_input("Heure de départ", value=datetime.strptime("09:00", "%H:%M").time())
day = st.date_input("Date de départ", value=date_end)  # Synchronize with sidebar date_end

# Calcul de l'itinéraire
if st.button("🚗 Calculer l'itinéraire"):
    with st.spinner("Optimisation en cours..."):
        timestamp = datetime.combine(day, hour)
        window = filtered_data[
            (filtered_data["timestamp_debut"] >= timestamp) &
            (filtered_data["timestamp_debut"] < timestamp + timedelta(minutes=30))
        ]
        if window.empty:
            st.error("Pas de données pour cette fenêtre.")
            st.stop()

        data = load_network_data()
        G = data["G"]
        noeuds = data["noeuds"]
        segments = data["segments"]

        path = optimize_route(G, start, end, None, segments)

        if path:
            st.success(f"Itinéraire : {' → '.join(path)}")
        else:
            st.error("Aucun itinéraire trouvé.")

# Méthodologie
with st.expander("📋 Méthodologie de Recherche"):
    st.markdown("""
    1. **Contexte** : Congestion croissante à Antananarivo
    2. **État de l'art** : ML + Graphes + Choix Discret
    3. **Méthodologie** : Données, modèles, validation
    4. **Résultats** : Interprétation et simulations
    5. **Conclusion** : Recommandations stratégiques
    """)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Plateforme d'optimisation des itinéraires - Antananarivo | Mémoire 2025</p>", unsafe_allow_html=True)