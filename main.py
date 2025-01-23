from fastapi import FastAPI
import uvicorn
from mlopspackage import data_loader, model_evaluation, model_training, preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

app = FastAPI()

# Add these lines where you define your app
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def read_root():
    with open("static/index.html", "r") as f:
        return f.read()


@app.get("/train")
def train_model():
    """
    Endpoint to train the model on the Titanic dataset.
    Returns a JSON containing model metrics.
    """
    # Charger les données
    train_data = data_loader.load_data('data/train.csv')
    train_data = data_loader.clean_data(train_data)

    # Préparer les données
    X = train_data.drop("Survived", axis=1)
    y = train_data["Survived"]
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.25, random_state=0
    )

    # Prétraitement
    categorical_features = ['Sex', 'Embarked', 'Cabin']
    numeric_features = ['Age', 'SibSp', 'Parch', 'Pclass', 'Fare']

    X_train_processed, X_valid_processed, preprocessor = preprocessing.preprocess_data(
        X_train, 
        X_valid, 
        numeric_features, 
        categorical_features
    )

    # Entraînement
    results, best_model, best_model_name = model_training.train_models(
        preprocessor, 
        X_train_processed, 
        y_train
    )

    # Évaluation
    classification_report = model_evaluation.evaluate_model(
        best_model, 
        X_valid_processed, 
        y_valid
    )

    # Préparer et renvoyer les résultats
    return {
        "results": {model: float(score) for model, score in results.items()},
        "best_model_name": best_model_name,
        "classification_report": str(classification_report)  # Convert to string
    }


if __name__ == "__main__":
    # Run the FastAPI app with uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
