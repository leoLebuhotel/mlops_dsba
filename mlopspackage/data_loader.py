import pandas as pd
import numpy as np

def load_data(path):
    """Charge les données depuis un fichier CSV."""
    return pd.read_csv(path)

def clean_data(data):
    """
       Nettoie les données en supprimant les colonnes inutiles et en gérant les valeurs manquantes.
       data: input file par l'utilisateur
    
    """
    # Suppression des colonnes inutiles
    data.drop(['Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)
    data["Cabin"] = data["Cabin"].apply(lambda s : s[0] if pd.notnull(s) else np.nan)

    # Gestion des valeurs manquantes
    num_cols = data.select_dtypes(include=[float, int]).columns
    cat_cols = data.select_dtypes(include=['object']).columns

    for col in num_cols:
        median_val = data[col].median()
        data[col] = data[col].fillna(median_val)

    for col in cat_cols:
        freq_val = data[col].mode()[0]
        data[col] = data[col].fillna(freq_val)

    return data