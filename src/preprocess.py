import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path):
    """
    Carga el dataset desde una ruta relativa o absoluta.
    Si la ruta es relativa, se resuelve con base a la raíz del proyecto.
    """
    # Ruta absoluta segura basada en la ubicación de este archivo
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_path = os.path.join(base_path, path)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"No se encontró el dataset en: {full_path}")

    df = pd.read_csv(full_path)
    df = df.dropna()

    X = df.drop("target", axis=1)
    y = df["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test
