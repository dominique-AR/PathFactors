## PathFactors 🧭

* Analyse (sans prise de tête) des facteurs qui influencent le choix des itinéraires, avec Streamlit pour tout voir en un clin d’œil.
* Objectif : trouver vite les critères qui rendent un trajet “optimal” (ou au moins pas trop nul) 😴✨

## 🧩 Description (court et net)

PathFactors permet de :

* Visualiser les facteurs clés (trafic, distance, préférences, etc.).
* Comparer des itinéraires selon des critères personnalisables.
* Détecter des tendances récurrentes dans les choix de déplacement.

## 📝 Principe :
tu donnes un CSV, l’app te montre des graphes/tableaux utiles, et tu ajustes tes critères. Basta.

## 🧱 Prérequis
- Python 3.8+
- pip
- (optionnel) environnement virtuel

## ⚙️ Installation
git clone https://github.com/dominique-AR/PathFactors.git
cd PathFactors
pip install -r requirements.txt

# (optionnel) créer/activer un venv
python -m venv .venv
.venv\Scripts\activate    # Windows
source .venv/bin/activate # macOS / Linux

## ▶️ Utilisation
streamlit run app.py

- Ouvrir le navigateur (souvent http://localhost:8501)
- Charger un CSV (ex. csv/data.csv)
- Choisir les facteurs à analyser
- Observer les graphiques et tableaux → ajuster → recommencer

## 🗺️ Diagramme du projet (mermaid, non interprété ici)
```plaintext
├── Accueil.py
├── pages/
│   ├── 1_Geospatial_Analysis.py
│   └── 2_XGBoost_SHAP_Analysis.py
└── utils/
    ├── data_loader.py
    ├── routing.py
    └── predictor.py


