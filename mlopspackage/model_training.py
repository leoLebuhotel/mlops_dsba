from sklearn.pipeline import Pipeline

def train_models(preprocessor, X_train, y_train):
    """
    Trains different models and returns their performance.

    Parameters:
    preprocessor (object): The preprocessing pipeline to be applied to the data.
    X_train (pd.DataFrame or np.ndarray): The training input samples.
    y_train (pd.Series or np.ndarray): The target values.

    Returns:
    dict: A dictionary where the keys are model names and the values are the mean cross-validated ROC AUC scores.
    """
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
            ('preprocessor', preprocessor),
            ('classifier', model) 
        ])
        scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='roc_auc')
        results[name] = scores.mean()
    return results