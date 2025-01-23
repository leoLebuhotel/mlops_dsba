from sklearn.metrics import classification_report

def evaluate_model(model, X_valid, y_valid):
    """Évalue un modèle sur un ensemble de validation."""
    y_pred = model.predict(X_valid)
    return classification_report(y_valid, y_pred)