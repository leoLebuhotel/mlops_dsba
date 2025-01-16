import pandas as pd
import numpy as np

def load_data(path):
    """Charge les données depuis un fichier CSV."""
    return pd.read_csv(path)

def clean_data(data):
    """Nettoie les données en supprimant les colonnes inutiles et en gérant les valeurs manquantes."""
    # Suppression des colonnes inutiles
<<<<<<< HEAD
    data.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)
=======
    data.drop(['Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)
    data["Cabin"] = data["Cabin"].apply(lambda s : s[0] if pd.notnull(s) else np.nan)
>>>>>>> d63520a4079a6a64e9b0296da65eb262f8c6aaf2

    # Gestion des valeurs manquantes
    num_cols = data.select_dtypes(include=[float, int]).columns
    cat_cols = data.select_dtypes(include=['object']).columns

    for col in num_cols:
        median_val = data[col].median()
<<<<<<< HEAD
        data[col].fillna(median_val)

    for col in cat_cols:
        freq_val = data[col].mode()[0]
        data[col].fillna(freq_val)
=======
        data[col] = data[col].fillna(median_val)

    for col in cat_cols:
        freq_val = data[col].mode()[0]
        data[col] = data[col].fillna(freq_val)
>>>>>>> d63520a4079a6a64e9b0296da65eb262f8c6aaf2

    return data