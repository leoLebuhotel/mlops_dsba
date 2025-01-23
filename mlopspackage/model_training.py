from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

def train_models(preprocessor, X_train, y_train):
    """
    Trains different models and returns their performance and the best model.

    Parameters:
    preprocessor (object): The preprocessing pipeline to be applied to the data.
    X_train (pd.DataFrame or np.ndarray): The training input samples.
    y_train (pd.Series or np.ndarray): The target values.

    Returns:
    dict: A dictionary where the keys are model names and the values are the mean cross-validated ROC AUC scores.
    object: The best model trained on the entire training set.
    """

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(eval_metric='logloss')
    }

    results = {}
    best_score = 0
    best_model = None
    best_model_name = None

    for name, model in models.items():
        pipe = Pipeline([
            ('classifier', model) 
        ])
        scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='roc_auc')
        mean_score = scores.mean()
        results[name] = mean_score

        if mean_score > best_score:
            best_score = mean_score
            best_model = pipe
            best_model_name = name

    # Train the best model on the entire training set
    best_model.fit(X_train, y_train)

    return results, best_model, best_model_name