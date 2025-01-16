from scripts.data_loader import load_data, clean_data
from scripts.preprocessing import preprocess_data
from scripts.model_training import train_models
from sklearn.model_selection import train_test_split

def main():
    # Charger les données
    train_data = load_data('data/train.csv')
    train_data = clean_data(train_data)

    # Préparer les données
    X = train_data.drop("Survived", axis=1)
    y = train_data["Survived"]

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=0)

    # Prétraitement
    numeric_features = ['Age', 'SibSp', 'Parch', 'Pclass', 'Fare']
    categorical_features = ['Sex', 'Embarked', 'Cabin']
    X_train_processed, X_valid_processed, preprocessor = preprocess_data(
        X_train, X_valid, numeric_features, categorical_features
    )

    # Entraînement
    results = train_models(preprocessor, X_train_processed, y_train)
    for model, score in results.items():
        print(f"{model}: {score:.2f}")

if __name__ == "__main__":
    main()