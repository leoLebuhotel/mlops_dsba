from mlopspackage import data_loader, model_evaluation, model_training, preprocessing
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    # Charger les données
    train_data = data_loader.load_data('data/train.csv')
    train_data = data_loader.clean_data(train_data)

    # Préparer les données
    X = train_data.drop("Survived", axis=1)
    y = train_data["Survived"]

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=0)

    # Prétraitement
    categorical_features = ['Sex', 'Embarked', 'Cabin']

    numeric_features = ['Age', 'SibSp', 'Parch', 'Pclass', 'Fare']
    X_train_processed, X_valid_processed, preprocessor = preprocessing.preprocess_data(
        X_train, X_valid, numeric_features, categorical_features
    )

    # Entraînement
    results,best_model, best_model_name = model_training.train_models(preprocessor, X_train_processed, y_train)
    for model, score in results.items():
        print(f"{model}: {score:.2f}")

    classification_report = model_evaluation.evaluate_model(best_model, X_valid_processed, y_valid)

    print(best_model_name) 
    print(classification_report)    

if __name__ == "__main__":
    main()