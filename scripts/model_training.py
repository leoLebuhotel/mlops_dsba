from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def train_models(preprocessor, X_train, y_train):
    """Entraîne différents modèles et retourne leurs performances."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }
    
    results = {}
    for name, model in models.items():
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy')
        results[name] = scores.mean()
    return results