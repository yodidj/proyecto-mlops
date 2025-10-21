# 🤖 Proyecto MLOps – Automatización de Pipeline de Machine Learning

## 🎯 Objetivo
Desarrollar un pipeline reproducible de machine learning que entrene, evalúe y registre un modelo automáticamente usando **GitHub Actions** y **MLflow**.

---

## 🧩 Estructura del Proyecto
```
proyecto-mlops/
├── src/
├── data/
├── mlruns/
└── .github/workflows/ml.yml
```

---

## 🚀 Instrucciones de uso

### 1️⃣ Clonar el repositorio
```bash
git clone https://github.com/tuusuario/proyecto-mlops.git
cd proyecto-mlops
```

### 2️⃣ Instalar dependencias
```bash
make install
```

### 3️⃣ Ejecutar entrenamiento
```bash
make train
```

### 4️⃣ Ver resultados en MLflow
```bash
mlflow ui
```
Accede en [http://localhost:5000](http://localhost:5000).

---

## ⚙️ CI/CD con GitHub Actions

El pipeline ejecuta automáticamente:
1. Instalación de dependencias  
2. Formato de código  
3. Entrenamiento del modelo  
4. Subida de artefactos (mlruns)

---

## 📊 Resultados esperados
- Accuracy y F1-score registrados en MLflow  
- Modelo guardado como artefacto  
- Ejecución automática en cada `push`
