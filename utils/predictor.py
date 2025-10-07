import numpy as np
import shap
import xgboost as xgb

def predict_saturation(features):
    model = xgb.XGBRegressor(
        max_depth=6, n_estimators=100, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    )

    if "precipitation_mmh" not in features.columns:
        features["precipitation_mmh"] = np.random.uniform(0, 5, len(features))
    if "poids_attractivite" not in features.columns:
        features["poids_attractivite"] = np.random.uniform(0.5, 1.5, len(features))

    X = features[["volume_total", "precipitation_mmh", "poids_attractivite"]].fillna(0)
    y = features["ratio_saturation"].fillna(0) + np.random.normal(0, 0.1, len(features))
    model.fit(X, y)
    predictions = model.predict(X)
    return predictions, model

def explain_predictions(features, model):
    explainer = shap.TreeExplainer(model)
    return explainer.shap_values(
        features[["volume_total", "precipitation_mmh", "poids_attractivite"]].fillna(0)
    )
