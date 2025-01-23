import pandas as pd

def preprocess_data(X_train, X_valid, numeric_features, categorical_features):
    """Prépare les données avec scaling et encodage des variables catégoriques."""
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ])
    preprocessor.fit(X_train)

    # Transformations
    X_train_transformed = preprocessor.transform(X_train)
    X_valid_transformed = preprocessor.transform(X_valid)

    # Reconvertir en DataFrame
    feature_names = (
        numeric_features +
        list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
    )
    X_train_df = pd.DataFrame(X_train_transformed, columns=feature_names, index=X_train.index)
    X_valid_df = pd.DataFrame(X_valid_transformed, columns=feature_names, index=X_valid.index)

    return X_train_df, X_valid_df, preprocessor