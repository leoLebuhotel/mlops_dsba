from scripts.data_loader import load_data, clean_data
from scripts.preprocessing import preprocess_data
from scripts.model_training import train_models
from sklearn.model_selection import train_test_split

def main():
    # Charger et nettoyer les données
    train_data = load_data('data/train.csv')
    train_data = clean_data(train_data)
    
    # Préparer les données pour l'entraînement
    X_train, X_valid, y_train, y_valid = train_test_split(
        train_data.drop("Survived", axis=1),
        train_data["Survived"],
        test_size=0.25,
        random_state=0
    )
    
    numeric_features = ['Age', 'SibSp', 'Parch', 'Fare']
    categorical_features = ['Sex', 'Embarked', 'Pclass']
    X_train_processed, X_valid_processed, preprocessor = preprocess_data(
        X_train, X_valid, numeric_features, categorical_features
    )
    
    # Entraîner les modèles
    results = train_models(preprocessor, X_train_processed, y_train)
    for model, score in results.items():
        print(f"{model}: {score:.2f}")

if __name__ == "__main__":
    main()