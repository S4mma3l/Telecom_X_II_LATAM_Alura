import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb # Importación de LightGBM
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Recargar el dataset
df = pd.read_csv('datos\\telecom_churn_clean.csv')
# Asegurarse de que la columna 'Churn' no tenga valores nulos
df.dropna(subset=['Churn'], inplace=True)

# 1. Separar las variables predictoras (X) de la variable objetivo (y)
X = df.drop('Churn', axis=1)
y = df['Churn']

# 2. Identificar columnas categóricas y numéricas
categorical_cols = X.select_dtypes(include='object').columns
numerical_cols = X.select_dtypes(include=np.number).columns

print(f"Columnas categóricas identificadas: {list(categorical_cols)}")
print(f"Columnas numéricas identificadas: {list(numerical_cols)}")

# 3. Crear el preprocesador usando ColumnTransformer
# Se aplicará StandardScaler a las columnas numéricas y OneHotEncoder a las categóricas
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# 4. Aplicar el preprocesamiento a X
X_processed = preprocessor.fit_transform(X)

# Convertir X_processed a un DataFrame para mejor visualización y manejo
feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
new_column_names = list(numerical_cols) + list(feature_names)
X_processed_df = pd.DataFrame(X_processed, columns=new_column_names)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_processed_df, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nDimensiones de X_train: {X_train.shape}")
print(f"Dimensiones de X_test: {X_test.shape}")
print(f"Dimensiones de y_train: {y_train.shape}")
print(f"Dimensiones de y_test: {y_test.shape}")

# Inicializar los modelos
models = {
    "Logistic Regression": LogisticRegression(random_state=42, solver='liblinear'),
    "Random Forest": RandomForestClassifier(random_state=42),
    "LightGBM": lgb.LGBMClassifier(random_state=42)
}

# Entrenar y evaluar cada modelo
trained_models = {}
cv_results = {}

for name, model in models.items():
    print(f"\n--- Entrenando y evaluando: {name} ---")

    # Entrenamiento del modelo
    model.fit(X_train, y_train)
    trained_models[name] = model
    print(f"{name} entrenado exitosamente.")

    # Evaluación con Cross-Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_processed_df, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    cv_results[name] = scores
    print(f"Resultados de Cross-Validation (ROC AUC): {scores.mean():.4f} +/- {scores.std():.4f}")

print("\n--- Entrenamiento de modelos completado ---")

# Los modelos entrenados están ahora en el diccionario 'trained_models'
# y sus resultados de validación cruzada en 'cv_results'.