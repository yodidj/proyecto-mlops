import mlflow
import mlflow.sklearn
import yaml
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from src.preprocess import load_data
from src.model import build_model


if __name__ == "__main__":
    # -----------------------------
    # Cargar configuración
    # -----------------------------
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Cargar datos
    X_train, X_test, y_train, y_test = load_data(config["data_path"])
    model = build_model(config["model_params"])

    # -----------------------------
    # Entrenamiento y evaluación
    # -----------------------------
    with mlflow.start_run(run_name="Entrenamiento_LogReg"):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # Métricas
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")
        precision = precision_score(y_test, preds, average="weighted", zero_division=0)
        recall = recall_score(y_test, preds, average="weighted", zero_division=0)

        # -----------------------------
        # Registrar métricas en MLflow
        # -----------------------------
        mlflow.log_params(config["model_params"])
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        # -----------------------------
        # Graficar y guardar matriz de confusión
        # -----------------------------
        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Matriz de Confusión")
        plt.xlabel("Predicción")
        plt.ylabel("Real")

        # Guardar la imagen localmente
        os.makedirs("results", exist_ok=True)
        cm_path = "results/confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()

        # Registrar imagen como artefacto
        mlflow.log_artifact(cm_path)

        # -----------------------------
        # Guardar modelo
        # -----------------------------
        mlflow.sklearn.log_model(model, artifact_path="model")

        # -----------------------------
        # Mostrar resultados en consola
        # -----------------------------
        print("\n✅ Entrenamiento finalizado con éxito:")
        print(f"Accuracy:  {acc:.3f}")
        print(f"F1 Score:  {f1:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall:    {recall:.3f}")
        print("\nMatriz de confusión guardada en:", cm_path)

