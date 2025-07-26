import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
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

# Entrenar los modelos
trained_models = {}
for name, model in models.items():
    print(f"\n--- Entrenando: {name} ---")
    model.fit(X_train, y_train)
    trained_models[name] = model
    print(f"{name} entrenado exitosamente.")

# Evaluación del rendimiento de los modelos con métricas
print("\n--- Evaluación del Rendimiento de los Modelos en el conjunto de Prueba ---")

evaluation_results = {}

for name, model in trained_models.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] # Probabilidades para la clase positiva (churn=1)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    evaluation_results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC AUC": roc_auc,
        "Confusion Matrix": cm
    }

    print(f"\n--- Métricas para {name} ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print("Matriz de Confusión:\n", cm)

    # Visualizar la Matriz de Confusión
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
    plt.title(f'Matriz de Confusión para {name}')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.savefig(f'img\\confusion_matrix_{name.replace(" ", "_")}.png')
    plt.show()

# Comparar todos los modelos basándose en ROC AUC
print("\n--- Comparación de Modelos (ROC AUC) ---")
for name, results in evaluation_results.items():
    print(f"{name}: ROC AUC = {results['ROC AUC']:.4f}")