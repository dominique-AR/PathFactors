import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set up the page
st.set_page_config(page_title="XGBoost & SHAP Analysis", layout="wide")

# Titre principal avec style
st.markdown("""
<div style="background: linear-gradient(135deg, #1f77b4, #4CAF50); padding: 20px; border-radius: 10px;">
    <h1 style="color: white; text-align: center;">Modélisation et Prédiction des Choix d'Itinéraires</h1>
    <h2 style="color: white; text-align: center;">Optimisation des Flux de Trafic à Antananarivo</h2>
</div>
""", unsafe_allow_html=True)

# Section d'explication en français avec onglets
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Aperçu", "🔄 Processus", "📥 Données d'entrée", "📤 Résultats", "✅ Validation"])

with tab1:
    st.markdown("""
    Cette application utilise l'apprentissage automatique pour prédire la saturation du trafic routier 
    (ratio volume/capacité) en utilisant le modèle XGBoost et l'analyse SHAP pour l'interprétabilité.
    """)

with tab2:
    st.markdown("""
    ### 🔄 Processus de modélisation:

    1. **📊 Chargement des données** - Données des segments routiers et profils de trafic
    2. **⚙️ Prétraitement** - Normalisation des variables numériques et encodage des variables catégorielles
    3. **⏱️ Validation temporelle** - Division des données avec TimeSeriesSplit pour éviter le data leakage
    4. **🎯 Entraînement du modèle** - Optimisation des hyperparamètres avec GridSearchCV
    5. **📈 Évaluation** - Calcul des métriques de performance (MAE, RMSE, R²)
    6. **🔍 Interprétation** - Analyse SHAP pour comprendre l'importance des variables
    """)

with tab3:
    st.markdown("""
    ### 📥 Données d'entrée:
    - **Segments routiers** : Caractéristiques physiques des routes (longueur, nombre de voies, etc.)
    - **Profils de trafic** : Données temporelles de volume et vitesse
    - **Variables temporelles** : Heure, jour de la semaine, mois
    """)

with tab4:
    st.markdown("""
    ### 📤 Résultats:
    - **Prédictions** : Ratio de saturation du trafic (volume/capacité)
    - **Métriques** : MAE, RMSE, R² pour évaluer la performance
    - **Importance des variables** : Analyse SHAP pour l'interprétabilité
    """)

with tab5:
    st.markdown("""
    ### ✅ Validation:
    - **Validation croisée temporelle** : Préservation de l'ordre chronologique
    - **Split train/test** : Séparation temporelle des données
    - **Optimisation des hyperparamètres** : Recherche par grille avec TimeSeriesSplit
    """)

# Load data from your CSV files
@st.cache_data
def load_data():
    # Load your actual data files
    segments_df = pd.read_csv('csv/segments.csv')
    segment_profiles_df = pd.read_csv('csv/segment_profiles.csv')
    predictions_df = pd.read_csv('csv/predictions_modele.csv')
    shap_values_df = pd.read_csv('csv/shap_values.csv')
    
    # Convert id_segment to the same type in both dataframes
    segments_df['id_segment'] = segments_df['id_segment'].astype(str)
    segment_profiles_df['id_segment'] = segment_profiles_df['id_segment'].astype(str)
    
    # Merge the data to create a comprehensive dataset
    df_main = pd.merge(segment_profiles_df, segments_df, on='id_segment', how='left')
    
    # Extract temporal features from timestamp
    df_main['timestamp_debut'] = pd.to_datetime(df_main['timestamp_debut'])
    df_main['hour'] = df_main['timestamp_debut'].dt.hour
    df_main['day_of_week'] = df_main['timestamp_debut'].dt.dayofweek
    df_main['month'] = df_main['timestamp_debut'].dt.month
    
    return df_main, predictions_df, shap_values_df

# Main function for this page
def main():
    # Load data
    df_main, predictions_df, shap_values_df = load_data()
    
    # Display the data
    if st.checkbox("📊 Afficher les données brutes"):
        st.subheader("Données des segments")
        st.dataframe(df_main.head())
        
        st.subheader("Données de prédictions")
        st.dataframe(predictions_df.head())
        
        st.subheader("Valeurs SHAP")
        st.dataframe(shap_values_df.head())
    
    # Display dataset size information
    st.sidebar.header("📋 Informations sur le dataset")
    st.sidebar.write(f"Échantillons totaux: {len(df_main)}")
    st.sidebar.write(f"Variables: {len(df_main.columns)}")
    
    # Model configuration
    st.sidebar.header("⚙️ Configuration du modèle")
    test_size = st.sidebar.slider("Taille du jeu de test", 0.1, 0.3, 0.2)
    
    # Dynamically adjust the maximum number of splits based on dataset size
    # Correction de l'erreur du slider
    max_possible_splits = max(2, min(10, len(df_main) - 2))  # Ensure at least 2 splits
    
    # Vérification que max_possible_splits est supérieur à 2
    if max_possible_splits > 2:
        n_splits = st.sidebar.slider("Nombre de splits (TimeSeriesSplit)", 2, max_possible_splits, min(5, max_possible_splits))
    else:
        n_splits = 2
        st.sidebar.write(f"Nombre de splits fixé à 2 (limite pour la taille du dataset)")
    
    # Définition de la variable cible
    target = 'ratio_saturation'
    
    # Séparation des features et de la target
    # Select features that might be predictive of traffic saturation
    feature_columns = [
        'volume_total', 'vitesse_moy_kmh', 'longueur_metres', 'nb_voies', 
        'vitesse_ref_kmh', 'pente_pourcentage', 'capacite_base_veh_h', 
        'hour', 'day_of_week', 'month', 'type_segment', 'sens_unique', 
        'voie_bus', 'restriction_pl'
    ]
    
    # Filter to only include columns that exist in the dataframe
    available_features = [col for col in feature_columns if col in df_main.columns]
    
    y = df_main[target]
    X = df_main[available_features]
    
    # Check if we have enough data for modeling
    if len(X) < 5:
        st.warning(f"❌ Données insuffisantes pour la modélisation. Seulement {len(X)} échantillons disponibles. Au moins 5 échantillons sont nécessaires.")
        return
    
    # Identification des types de colonnes
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    # Afficher les variables utilisées
    st.sidebar.subheader("🎯 Variables utilisées")
    st.sidebar.write("**Numériques:**", ", ".join(numeric_features))
    st.sidebar.write("**Catégorielles:**", ", ".join(categorical_features))
    st.sidebar.write("**Cible:**", target)
    
    # Création du préprocesseur
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Split des données avec validation temporelle
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False, random_state=42)
    
    # Check if we have enough training data for the selected number of splits
    if len(X_train) <= n_splits:
        st.warning(f"⚠️ Échantillons d'entraînement insuffisants ({len(X_train)}) pour {n_splits} splits. Réduction à {len(X_train)-1} splits.")
        n_splits = max(2, len(X_train) - 1)
    
    # Configuration de la validation croisée temporelle
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Pipeline intégrant le préprocessing et l'estimateur XGBoost
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', xgb.XGBRegressor(objective='reg:squarederror', random_state=42))
    ])
    
    # Définition de la grille d'hyperparamètres à tester
    param_grid = {
        'regressor__max_depth': [3, 4],
        'regressor__n_estimators': [50, 100],
        'regressor__learning_rate': [0.01, 0.1],
        'regressor__subsample': [0.8, 1.0],
        'regressor__colsample_bytree': [0.8, 1.0]
    }
    
    # Simplify the parameter grid if we have a small dataset
    if len(X_train) < 20:
        param_grid = {
            'regressor__max_depth': [3],
            'regressor__n_estimators': [50],
            'regressor__learning_rate': [0.1],
        }
    
    # Train model button
    if st.button("🚀 Entraîner le modèle"):
        st.subheader("📈 Progression de l'entraînement du modèle")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Mise en place de la recherche par grille
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=tscv,
            scoring='neg_root_mean_squared_error',
            verbose=1,
            n_jobs=-1
        )
        
        status_text.text("🔍 Démarrage de la recherche par grille...")
        try:
            grid_search.fit(X_train, y_train)
            progress_bar.progress(50)
            
            status_text.text("✅ Recherche par grille terminée! Évaluation du modèle...")
            
            # Récupération du meilleur estimateur
            best_model = grid_search.best_estimator_
            
            # Prédictions sur le jeu de test
            y_pred = best_model.predict(X_test)
            
            # Calcul des métriques
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            progress_bar.progress(100)
            status_text.text("🎉 Évaluation terminée!")
            
            # Display results
            st.subheader("📊 Performance du modèle")
            col1, col2, col3 = st.columns(3)
            col1.metric("MAE (Erreur Absolue Moyenne)", f"{mae:.4f}")
            col2.metric("RMSE (Racine de l'Erreur Quadratique Moyenne)", f"{rmse:.4f}")
            col3.metric("R² (Coefficient de Détermination)", f"{r2:.4f}")
            
            st.subheader("⚙️ Meilleurs paramètres")
            st.write(grid_search.best_params_)
            
            # Visualisation des prédictions vs. observations réelles
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(y_test, y_pred, alpha=0.5)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
            ax.set_xlabel('Valeur Observée (ratio_saturation)')
            ax.set_ylabel('Valeur Prédite (ratio_saturation)')
            ax.set_title('XGBoost : Prédictions vs. Observations Réelles')
            st.pyplot(fig)
            
            # SHAP Analysis (only if we have enough data)
            if len(X_train) > 10:
                st.subheader("🔍 Analyse SHAP")
                st.markdown("""
                L'analyse SHAP (SHapley Additive exPlanations) permet de comprendre l'importance 
                de chaque variable dans les prédictions du modèle. Les valeurs SHAP montrent 
                comment chaque caractéristique contribue à la prédiction pour chaque instance.
                """)
                
                # Initialisation de l'explainer SHAP
                preprocessor = best_model.named_steps['preprocessor']
                xgb_regressor = best_model.named_steps['regressor']
                
                # Application du préprocessing aux données d'entraînement
                X_train_preprocessed = preprocessor.transform(X_train)
                
                # Création d'un explainer SHAP
                explainer = shap.TreeExplainer(xgb_regressor)
                
                # Calcul des valeurs SHAP pour un échantillon
                sample_size = min(100, X_train_preprocessed.shape[0])
                shap_values = explainer.shap_values(X_train_preprocessed[:sample_size])
                
                # Obtention des noms de features après preprocessing
                feature_names = numeric_features.copy()
                ohe_categories = preprocessor.named_transformers_['cat'].categories_
                for i, cats in enumerate(ohe_categories):
                    feature_names.extend([f"{categorical_features[i]}_{cat}" for cat in cats])
                
                # Summary plot global
                st.write("📊 Importance globale des variables (SHAP)")
                fig, ax = plt.subplots(figsize=(10, 8))
                shap.summary_plot(shap_values, X_train_preprocessed[:sample_size], 
                                  feature_names=feature_names, plot_type="bar", show=False)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Summary plot détaillé (beeswarm)
                st.write("🎯 Impact et direction des variables (SHAP)")
                fig, ax = plt.subplots(figsize=(10, 8))
                shap.summary_plot(shap_values, X_train_preprocessed[:sample_size], 
                                  feature_names=feature_names, show=False)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Légende pour l'analyse SHAP
                st.markdown("""
                #### 📝 Légende de l'analyse SHAP:
                - **Valeur SHAP** : Impact sur la prédiction (positif ou négatif)
                - **Couleur** : Valeur de la variable (rouge = élevée, bleu = basse)
                - **Position horizontale** : Contribution à la prédiction
                - **Ordre vertical** : Importance des variables (de haut en bas)
                """)
            else:
                st.info("ℹ️ Données insuffisantes pour l'analyse SHAP. Au moins 10 échantillons sont nécessaires.")
                
        except Exception as e:
            st.error(f"❌ Une erreur s'est produite pendant l'entraînement du modèle: {str(e)}")
        
        # Compare with your existing SHAP values
        st.subheader("🔁 Comparaison avec les valeurs SHAP précalculées")
        st.dataframe(shap_values_df.head(10))

# Run the app
if __name__ == "__main__":
    main()