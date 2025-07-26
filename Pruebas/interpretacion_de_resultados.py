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

# Recargar el dataset y realizar el preprocesamiento completo (replicando los pasos anteriores)
df = pd.read_csv('datos\\telecom_churn_clean.csv')
df.dropna(subset=['Churn'], inplace=True)

X = df.drop('Churn', axis=1)
y = df['Churn']
categorical_cols = X.select_dtypes(include='object').columns
numerical_cols = X.select_dtypes(include=np.number).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

X_processed = preprocessor.fit_transform(X)
feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
new_column_names = list(numerical_cols) + list(feature_names)
X_processed_df = pd.DataFrame(X_processed, columns=new_column_names)

X_train, X_test, y_train, y_test = train_test_split(X_processed_df, y, test_size=0.2, random_state=42, stratify=y)

# Re-entrenar los modelos (ya que trained_models no se persisten entre ejecuciones del interprete)
models = {
    "Logistic Regression": LogisticRegression(random_state=42, solver='liblinear'),
    "Random Forest": RandomForestClassifier(random_state=42),
    "LightGBM": lgb.LGBMClassifier(random_state=42)
}

trained_models = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[name] = model

# --- Interpretación de Resultados: Importancia de las Variables ---
print("\n--- Interpretación de Resultados: Importancia de las Variables ---")

# Obtener nombres de las características para la interpretación
feature_names_list = X_processed_df.columns.tolist()

# Función para visualizar la importancia de las características
def plot_feature_importance(model, feature_names, model_name, top_n=15):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_') and model_name == "Logistic Regression":
        # Para regresión logística, usamos los coeficientes absolutos como proxy de importancia
        importances = np.abs(model.coef_[0])
    else:
        print(f"El modelo {model_name} no tiene el atributo feature_importances_ ni coef_.")
        return

    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 7))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(top_n), palette='viridis')
    plt.title(f'Top {top_n} Importancia de Características para {model_name}')
    plt.xlabel('Importancia')
    plt.ylabel('Característica')
    plt.tight_layout()
    plt.savefig(f'img\\feature_importance_{model_name.replace(" ", "_")}.png')
    plt.show()

# Visualizar la importancia de las características para Random Forest y LightGBM
plot_feature_importance(trained_models["Random Forest"], feature_names_list, "Random Forest")
plot_feature_importance(trained_models["LightGBM"], feature_names_list, "LightGBM")

# Opcional: Para Regresión Logística, podemos interpretar los coeficientes (aunque no es una "importancia" directa como en árboles)
# Coeficientes de Regresión Logística (magnitud)
log_reg_coefs = pd.DataFrame({'Feature': feature_names_list, 'Coefficient': trained_models["Logistic Regression"].coef_[0]})
log_reg_coefs['Absolute_Coefficient'] = np.abs(log_reg_coefs['Coefficient'])
log_reg_coefs = log_reg_coefs.sort_values(by='Absolute_Coefficient', ascending=False)

print("\n--- Top 15 Coeficientes (Magnitud) para Regresión Logística ---")
print(log_reg_coefs.head(15))