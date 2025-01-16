from sklearn.pipeline import Pipeline

def train_models(preprocessor, X_train, y_train):
    """Entraîne différents modèles et retourne leurs performances."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.model_selection import cross_val_score

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(eval_metric='logloss')
    }

    results = {}
    for name, model in models.items():
        pipe = Pipeline([
            ('classifier', model) 
        ])
        scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='roc_auc')
        results[name] = scores.mean()
    return results