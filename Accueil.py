
import streamlit as st

st.set_page_config(page_title="Mon Application", layout="wide")

# Titre principal
st.markdown("""
<div style="background: linear-gradient(135deg, #1f77b4, #4CAF50); padding: 20px; border-radius: 10px;">
    <h1 style="color: white; text-align: center;">Modélisation et Prédiction des Choix d'Itinéraires</h1>
    <h2 style="color: white; text-align: center;">Optimisation des Flux de Trafic à Antananarivo</h2>
</div>
""", unsafe_allow_html=True)


# Cadre théorique
tab1, tab2, tab3, tab4 = st.tabs(["📘 Cadre Théorique", "❓ Hypothèses", "📍 Objectifs", "⚙️ Outils principaux"])

with tab1:
    st.markdown("""
    ### 📘 Cadre Théorique et Conceptuel

    #### 🎯 Pourquoi cette étude ?
    Cette recherche vise à modéliser et prédire les choix d'itinéraires des conducteurs dans un contexte urbain hétérogène, 
    afin de mieux comprendre les facteurs de saturation et proposer des solutions de contrôle efficaces pour réduire la congestion 
    à Antananarivo, une ville confrontée à des défis croissants de mobilité.

    #### 🔍 Comment abordons-nous le problème ?
    Notre approche combine trois disciplines complémentaires:
    - **Économie des transports** : Modèles de choix discret (Logit, Mixed Logit) pour relier les coûts généralisés aux décisions des usagers
    - **Théorie des graphes** : Algorithmes de plus court chemin (Dijkstra, A*) pour la représentation du réseau et l'identification d'alternatives
    - **Apprentissage automatique** : Algorithmes comme XGBoost pour capter les interactions non linéaires entre variables
    """)

with tab2:
    st.markdown("""
    ### ❓ Hypothèses

    1. Les choix d'itinéraires dépendent principalement des coûts généralisés (temps de parcours, saturation volume-capacité, fiabilité)
    2. Les typologies de trajets (urbains, périurbains, transit) influencent fortement les décisions des usagers
    3. L'intégration d'un modèle prédictif (ML) avec des méthodes de graphes améliore la précision des estimations de saturation
    """)

with tab3:
    st.markdown("""
    ### 📍 Objectifs

    - Construire un système de traitement de données hétérogènes (cartographie, GPS, comptages trafic, variables exogènes)
    - Classifier les itinéraires selon leur typologie et identifier les nœuds critiques du réseau
    - Développer un modèle prédictif pour estimer les durées de trajet et niveaux de congestion
    - Analyser l'influence des facteurs déterminants avec des outils d'interprétabilité
    - Proposer des scénarios opérationnels de régulation réalistes
    """)

with tab4:
    st.markdown("""
    #### ⚙️ Outils principaux
    - Modélisation prédictive: XGBoost pour l'estimation des durées de trajet et niveaux de congestion
    - Analyse de réseau : Représentation graphique du réseau routier et identification des nœuds critiques
    - Interprétabilité : SHAP values pour comprendre l'importance des variables dans les prédictions
    """)