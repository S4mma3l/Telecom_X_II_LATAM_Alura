import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import warnings

# Ignore warnings for a cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)

# --- 1. Carga de Datos ---
print("--- 1. Carga de Datos ---")
try:
    file_path = 'datos\\telecom_churn_clean.csv'
    df = pd.read_csv(file_path)
    print(f"DataFrame '{file_path}' cargado. Dimensiones iniciales: {df.shape}")
except FileNotFoundError:
    print(f"Error Crítico: El archivo '{file_path}' no se encontró.")
    raise

# --- 2. Preprocesamiento Correcto ---
print("\n--- 2. Preprocesamiento Correcto ---")

df.columns = [col.replace('.', '_').strip() for col in df.columns]

# The 'Churn' column is numeric. The only step needed is to handle potential nulls.
df.dropna(subset=['Churn'], inplace=True)
df['Churn'] = df['Churn'].astype(int)

# Handle nulls in 'account_Charges_Total'
df['account_Charges_Total'] = pd.to_numeric(df['account_Charges_Total'], errors='coerce')
df['account_Charges_Total'].fillna(df['account_Charges_Total'].median(), inplace=True)

print(f"Dimensiones tras limpieza: {df.shape}")

# --- 3. Preparación para el Modelo ---
print("\n--- 3. Preparando Datos para el Modelo ---")
X = df.drop('Churn', axis=1)
y = df['Churn']

# Identify feature types
categorical_features = X.select_dtypes(include=['object', 'category']).columns
numerical_features = X.select_dtypes(include=np.number).columns
print(f"Características identificadas: {len(numerical_features)} numéricas y {len(categorical_features)} categóricas.")

# ### LA CORRECCIÓN CLAVE ESTÁ AQUÍ ###
# Create the preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        # This tuple now correctly includes the 'categorical_features' list
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)
# ### FIN DE LA CORRECCIÓN ###

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Datos divididos en {len(X_train)} para entrenamiento y {len(X_test)} para prueba.")

# --- 4. Entrenamiento, Evaluación y Optimización ---
print("\n--- 4. Entrenando, Evaluando y Optimizando ---")

models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42)
}

param_grid = {
    "Logistic Regression": {'classifier__C': [0.1, 1, 10]},
    "Random Forest": {'classifier__n_estimators': [100, 150], 'classifier__max_depth': [10, 20]}
}

model_performance = {}
optimized_models = {}
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_model(name, model, X_test_data, y_test_data):
    """Calculates and displays model metrics."""
    y_pred = model.predict(X_test_data)
    y_proba = model.predict_proba(X_test_data)[:, 1]
    
    precision = precision_score(y_test_data, y_pred, zero_division=0)
    recall = recall_score(y_test_data, y_pred, zero_division=0)
    f1 = f1_score(y_test_data, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test_data, y_proba)
    
    print(f"--- Métricas para {name} ---")
    print(f"  Accuracy:  {accuracy_score(y_test_data, y_pred):.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")
    return roc_auc

for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    
    print(f"\nOptimizando {name}...")
    grid_search = GridSearchCV(pipeline, param_grid[name], cv=cv_strategy, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    optimized_models[name] = grid_search.best_estimator_
    print(f"  Mejores parámetros: {grid_search.best_params_}")
    
    auc_score = evaluate_model(f"Optimizada - {name}", grid_search.best_estimator_, X_test, y_test)
    model_performance[name] = auc_score

# --- 5. Almacenamiento del Modelo Ganador ---
print("\n--- 5. Guardando el Modelo Ganador ---")

if model_performance:
    best_model_name = max(model_performance, key=model_performance.get)
    best_model_pipeline = optimized_models[best_model_name]
    best_model_score = model_performance[best_model_name]

    print(f"🏆 Modelo ganador: '{best_model_name}' (ROC-AUC: {best_model_score:.4f}).")

    model_filename = 'modelo\\mejor_modelo_churn.joblib'
    joblib.dump(best_model_pipeline, model_filename)
    print(f"✅ Modelo guardado exitosamente como '{model_filename}'")
else:
    print("❌ No se pudo determinar un modelo ganador.")