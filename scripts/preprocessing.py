from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def preprocess_data(X_train, X_valid, numeric_features, categorical_features):
    """Prépare les données avec scaling et encodage des variables catégoriques."""
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ])
    preprocessor.fit(X_train)
    return (
        preprocessor.transform(X_train),
        preprocessor.transform(X_valid),
        preprocessor
    )