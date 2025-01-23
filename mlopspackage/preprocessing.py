import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def preprocess_data(X_train, X_valid, numeric_features, categorical_features):
    """
    Preprocess the training and validation data by scaling numeric features and encoding categorical features.
    Parameters:
    X_train (pd.DataFrame): Training data.
    X_valid (pd.DataFrame): Validation data.
    numeric_features (list of str): List of numeric feature names.
    categorical_features (list of str): List of categorical feature names.
    Returns:
    pd.DataFrame: Transformed training data.
    pd.DataFrame: Transformed validation data.
    ColumnTransformer: Fitted preprocessor object.
    """

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