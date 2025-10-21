import mlflow
import mlflow.sklearn
import yaml
from sklearn.metrics import accuracy_score, f1_score
from src.preprocess import load_data
from src.model import build_model

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    X_train, X_test, y_train, y_test = load_data(config["data_path"])
    model = build_model(config["model_params"])

    with mlflow.start_run(run_name="Entrenamiento_LogReg"):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        mlflow.log_params(config["model_params"])
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"✅ Entrenamiento finalizado. Accuracy: {acc:.3f} | F1: {f1:.3f}")
