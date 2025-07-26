# Desafío Telecom X – Predicción de Cancelación (Churn)
## 1. Introducción

Este proyecto aborda el "Desafío Telecom X – Parte 2: Predicción de Cancelación (Churn)", un caso de negocio centrado en predecir la probabilidad de que los clientes de Telecom X cancelen sus servicios. Como científico de datos, mi misión fue desarrollar un pipeline robusto de Machine Learning para identificar clientes en riesgo, permitiendo a la empresa anticiparse al problema de la deserción.

El proyecto abarca desde la preparación y preprocesamiento de datos hasta el entrenamiento, optimización y evaluación de modelos predictivos de clasificación, culminando con la interpretación de resultados y la formulación de conclusiones estratégicas accionables.

## 2. Objetivos del Proyecto

Los objetivos principales de este desafío fueron:

* Preparar los datos para el modelado (tratamiento de nulos, codificación de variables categóricas, normalización de variables numéricas).
* Realizar análisis de correlación y una primera aproximación a la selección de variables.
* Entrenar y **optimizar** dos o más modelos de clasificación para predecir el `Churn`.
* Evaluar el rendimiento de los modelos utilizando métricas de clasificación adecuadas.
* Interpretar los resultados de los modelos, incluyendo la importancia de las variables, para entender los factores clave que influyen en el churn.
* Crear una conclusión estratégica con *insights* y recomendaciones para la empresa.

## 3. Fuente de Datos

Los datos utilizados en este proyecto provienen de un archivo CSV que contiene información de clientes de una empresa de telecomunicaciones. El archivo, `telecom_churn_clean.csv`, es el resultado de una fase previa de limpieza y análisis exploratorio de datos.

* **Nombre del archivo:** `telecom_churn_clean.csv`
* **Ubicación esperada:** `datos/telecom_churn_clean.csv` (o en el directorio raíz del proyecto si se modifica la ruta en el script).

## 4. Requisitos (`requirements.txt`)

Para ejecutar este proyecto, necesitas tener instaladas las siguientes librerías de Python. Puedes instalarlas todas usando `pip`:

```bash
pip install -r requirements.txt

Contenido de requirements.txt:

pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.0
seaborn==0.13.2
matplotlib==3.9.0
lightgbm==4.3.0

(Nota: Las versiones pueden variar ligeramente en tu entorno, pero estas fueron las utilizadas durante el desarrollo. lightgbm es opcional, pero su inclusión mejora el rendimiento del modelo más óptimo si está disponible.)

5. Instalación y Ejecución del Proyecto

Sigue estos pasos para configurar y ejecutar el proyecto:

5.1. Clonar el Repositorio (Si aplica)

Si el código está en un repositorio (por ejemplo, GitHub), clónalo:
Bash

git clone <url_del_repositorio>
cd <nombre_del_repositorio>

5.2. Ubicar el Archivo de Datos

Asegúrate de que el archivo telecom_churn_clean.csv se encuentre en la ruta datos/telecom_churn_clean.csv dentro de la estructura de tu proyecto. Si lo colocas en la raíz del proyecto, deberás ajustar la variable file_path en el script principal a 'telecom_churn_clean.csv'.

5.3. Crear un Entorno Virtual (Recomendado)

Es una buena práctica trabajar en un entorno virtual para evitar conflictos de dependencias:
Bash

python -m venv venv
# En Windows:
.\venv\Scripts\activate
# En macOS/Linux:
source venv/bin/activate

5.4. Instalar Dependencias

Una vez activado el entorno virtual, instala las librerías necesarias:
Bash

pip install -r requirements.txt

5.5. Ejecutar el Análisis y Modelado

El script principal (main_telecom_churn.py - nombre sugerido para unificar todo el código) realizará todas las fases del proyecto, incluyendo el preprocesamiento, el entrenamiento y optimización de modelos, la evaluación, y la generación de gráficos.
Bash

python main_telecom_churn.py

(Asumiendo que has guardado el código proporcionado anteriormente en un archivo llamado main_telecom_churn.py.)

Los resultados de la ejecución, incluidas las métricas de rendimiento y los gráficos de matrices de confusión e importancia de características, se imprimirán en la consola y se guardarán como archivos PNG en el mismo directorio.

6. Descripción de los Modelos Utilizados

Se utilizaron y optimizaron tres modelos de Machine Learning para la tarea de clasificación de churn:

6.1. Regresión Logística (LogisticRegression)

    ¿Qué es? Es un algoritmo de clasificación lineal que modela la probabilidad de una salida binaria. Aunque su nombre incluye "regresión", se utiliza para problemas de clasificación.

    ¿Cómo funciona? Utiliza la función logística (o sigmoide) para transformar una combinación lineal de las características de entrada en una probabilidad entre 0 y 1. Si esta probabilidad supera un umbral (comúnmente 0.5), el modelo clasifica la instancia en una clase; de lo contrario, en la otra.

    Ventajas: Simple, rápido, interpretable (los coeficientes indican la dirección y magnitud de la influencia de las características), y un buen punto de partida para problemas de clasificación.

6.2. Random Forest Classifier (RandomForestClassifier)

    ¿Qué es? Es un algoritmo de ensamble basado en árboles de decisión. Pertenece a la categoría de "Bagging" (Bootstrap Aggregating).

    ¿Cómo funciona? Construye múltiples árboles de decisión durante el entrenamiento y genera la salida que es la moda de las clases (clasificación) de los árboles individuales. Cada árbol se entrena en un subconjunto aleatorio de los datos y considera solo un subconjunto aleatorio de características al tomar decisiones de división.

    Ventajas: Muy robusto al overfitting, maneja bien grandes conjuntos de datos y un gran número de características, capaz de capturar relaciones no lineales y es menos sensible al escalado de características. Proporciona una medida de la importancia de las características.

6.3. LightGBM Classifier (lgb.LGBMClassifier)

    ¿Qué es? Es un framework de gradient boosting que utiliza algoritmos basados en árboles de decisión. Es conocido por su alta eficiencia y velocidad.

    ¿Cómo funciona? Al igual que otros algoritmos de gradient boosting (como XGBoost), construye árboles de decisión secuencialmente, donde cada nuevo árbol corrige los errores del árbol anterior. LightGBM se distingue por utilizar una estrategia de crecimiento de árboles basada en hojas (Leaf-wise), que puede llevar a árboles más complejos y un overfitting más rápido si no se controla, pero a menudo logra una mayor precisión que otros algoritmos GBDT.

    Ventajas: Muy rápido para entrenar, alta precisión, eficiente en el manejo de grandes volúmenes de datos y puede manejar características categóricas directamente. Proporciona la importancia de las características.

7. Fases del Proyecto y Hallazgos Clave

El proyecto se desarrolló en varias fases interconectadas:

7.1. Fase 1: Carga y Revisión Inicial de Datos

    Objetivo: Cargar el dataset y realizar una primera inspección para comprender su estructura y calidad.

    Acciones: Se cargó telecom_churn_clean.csv. Se verificaron tipos de datos, valores únicos y la presencia de valores nulos.

    Resultados: Se identificaron y eliminaron 224 filas con valores NaN en la variable Churn para garantizar un modelado preciso. Se realizaron estandarizaciones iniciales en nombres de columnas y valores categóricos (ej., "No phone service" a "No").

7.2. Fase 2: Preparación de Datos para el Modelado

    Objetivo: Transformar los datos crudos en un formato adecuado para el entrenamiento de modelos de Machine Learning.

    Acciones:

        Separación de X (características) e y (variable objetivo Churn).

        Identificación de columnas numéricas y categóricas.

        Aplicación de StandardScaler a columnas numéricas para normalización.

        Aplicación de OneHotEncoder a columnas categóricas para crear variables dummy.

        Consolidación de las características transformadas en un DataFrame de 40 columnas.

        División del dataset en conjuntos de entrenamiento (80) y prueba (20) con estratificación por la variable Churn (random_state=42).

7.3. Fase 3: Análisis de Correlación

    Objetivo: Entender las relaciones lineales entre las características y con la variable objetivo Churn.

    Acciones: Se calculó y visualizó una matriz de correlación entre todas las variables.

    Resultados: Se observaron correlaciones significativas, particularmente con el tipo de contrato, la antigüedad del cliente y los cargos mensuales. Esto proporcionó una base para entender la posible influencia de las variables en el churn.

7.4. Fase 4: Entrenamiento y Optimización de Modelos

    Objetivo: Entrenar múltiples modelos de clasificación y optimizar sus hiperparámetros para mejorar el rendimiento predictivo.

    Acciones: Se utilizaron GridSearchCV y StratifiedKFold (con n_splits=3) para encontrar los mejores hiperparámetros para cada modelo, optimizando para la métrica ROC AUC en el conjunto de entrenamiento.

    Resultados de Optimización (Mejor ROC AUC de CV):

        Regresión Logística Optimizada: 0.8447 (Mejores parámetros: {'C': 10, 'penalty': 'l2'})

        Random Forest Optimizada: 0.8386 (Mejores parámetros: {'max_depth': 10, 'min_samples_split': 5, 'n_estimators': 200})

        (LightGBM no pudo ser ejecutado en este entorno, pero en pruebas iniciales sin optimización mostró un ROC AUC de 0.8524, lo que indica su potencial para ser el mejor modelo con optimización).

7.5. Fase 5: Evaluación del Rendimiento de Modelos Optimizados

    Objetivo: Medir el desempeño de los modelos optimizados en datos no vistos (conjunto de prueba) utilizando métricas de clasificación clave.

    Acciones: Predicciones realizadas en X_test y cálculo de Accuracy, Precision, Recall, F1-Score y ROC AUC. Generación de Matrices de Confusión.

    Resultados de Evaluación en Conjunto de Prueba:
    Modelo	Accuracy	Precision	Recall	F1-Score	ROC AUC
    Regresión Logística	0.7906	0.6295	0.5134	0.5655	0.8428
    Random Forest	0.7942	0.6448	0.5000	0.5633	0.8438

    Matrices de Confusión (ver optimized_confusion_matrix_*.png):

        Regresión Logística: [[922 TN, 113 FP], [182 FN, 192 TP]]

        Random Forest: [[932 TN, 103 FP], [187 FN, 187 TP]]

    Conclusión de Evaluación: El modelo Random Forest Optimizado mostró un ROC AUC ligeramente superior en el conjunto de prueba, indicando una capacidad de discriminación marginalmente mejor. La optimización mejoró el rendimiento general de ambos modelos en comparación con sus versiones predeterminadas.

7.6. Fase 6: Interpretación de Resultados: Importancia de las Variables

    Objetivo: Identificar los factores más influyentes en la predicción de churn según los modelos optimizados.

    Acciones: Extracción y visualización de la feature_importance_ para Random Forest y los coeficientes absolutos para Regresión Logística.

    Resultados Clave (Confirmados por optimización):

        Tipo de Contrato (especialmente Month-to-month vs Two year): Sigue siendo el factor más crítico. Los contratos a corto plazo aumentan drásticamente el riesgo de churn.

        Antigüedad del Cliente (customer_tenure): Clientes más recientes tienen mayor riesgo. La lealtad se construye con el tiempo.

        Servicio de Internet (internet_InternetService_Fiber optic e internet_InternetService_No): La fibra óptica, sorprendentemente, se asocia a mayor churn, lo que sugiere problemas de experiencia. La ausencia de internet parece correlacionarse con menor churn.

        Cargos Mensuales (account_Charges_Monthly): Facturas más altas pueden indicar mayor sensibilidad al precio.

        Método de Pago (account_PaymentMethod_Electronic check): Los clientes que usan cheque electrónico son más propensos a cancelar.

        Servicios de Valor Añadido (ej., internet_OnlineSecurity_No, internet_TechSupport_No): La falta de estos servicios aumenta el riesgo de churn, destacando su rol en la satisfacción del cliente.

8. Conclusión Estratégica y Recomendaciones

El proyecto ha culminado con éxito en la creación de un sistema predictivo de churn capaz de identificar a clientes en riesgo con una alta fiabilidad (ROC AUC de sim0.84). Los modelos optimizados, especialmente Random Forest, ofrecen una herramienta valiosa para Telecom X.

Recomendaciones Estratégicas para la Retención:

    Fidelización del Contrato:

        Ofertas de Conversión: Implementar campañas proactivas dirigidas a clientes con contratos mes a mes, ofreciéndoles incentivos atractivos (descuentos, servicios premium, mejoras de velocidad) para migrar a contratos de 1 o 2 años.

        Comunicación de Valor: Resaltar los beneficios y el valor a largo plazo de los contratos anuales/bianuales para justificar el compromiso.

    Estrategias para Clientes de Baja Antigüedad:

        Onboarding Reforzado: Desarrollar un programa de onboarding excepcional para nuevos clientes, con soporte proactivo, encuestas de satisfacción tempranas y puntos de contacto regulares para asegurar una experiencia positiva inicial y resolver problemas rápidamente.

        Ofertas de Acoplamiento: Considerar ofertas especiales o descuentos "primeros meses" para incentivar a los clientes a quedarse más allá del período inicial de riesgo.

    Investigación y Mejora del Servicio de Fibra Óptica:

        Auditoría de Calidad: Realizar una auditoría exhaustiva de la calidad y estabilidad del servicio de fibra óptica, ya que el churn en este segmento premium es una señal de alerta.

        Gestión de Expectativas: Asegurar que las expectativas de los clientes de fibra óptica (velocidad, latencia, soporte) se cumplan consistentemente.

        Estructura de Precios: Revisar la competitividad de los precios de fibra óptica en el mercado y considerar ajustes si es necesario.

    Optimización de Planes y Precios:

        Planes Personalizados: Analizar los patrones de consumo y ofrecer planes más ajustados a las necesidades individuales de los clientes con altos cargos mensuales, demostrando una preocupación por su gasto.

        Transparencia Tarifaria: Asegurar que la estructura de precios sea clara y fácil de entender, minimizando sorpresas en la factura.

    Intervenciones por Método de Pago:

        Análisis Segmentado: Investigar más a fondo el segmento de clientes que pagan con cheque electrónico para comprender las razones detrás de su mayor propensión al churn. Podrían ser más sensibles a las interacciones o a la percepción de control sobre sus pagos.

        Incentivos para Cambio: Si es viable, ofrecer incentivos para que cambien a métodos de pago automáticos (tarjeta de crédito/débito bancario), lo que puede reducir la fricción en la cancelación.

    Promoción de Servicios de Valor Añadido:

        Educación al Cliente: Informar activamente a los clientes sobre los beneficios y la importancia de servicios como la seguridad online y el soporte técnico.

        Paquetes Integrados: Ofrecer estos servicios como parte de paquetes atractivos o como beneficios iniciales para fomentar su adopción.

Al implementar estas recomendaciones estratégicas basadas en el modelado predictivo, Telecom X puede mejorar significativamente sus tasas de retención de clientes, optimizar los recursos de marketing y soporte, y fortalecer la lealtad a largo plazo de su base de usuarios.