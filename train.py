# train.py

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# --- 1. Cargar datos ---
df = pd.read_csv("filtered_data.csv")  # Cambia por tu CSV filtrado

# --- 2. Selección de variables ---
num_cols = ['price', 'initial_quantity', 'sold_quantity']  # Ejemplo de numéricas seleccionadas
cat_cols = ['condition', 'buying_mode', 'listing_type_id']  # Ejemplo de categóricas limpias

# Convertir target a 0/1
y = df['condition'].map({'new': 0, 'used': 1})
X = df[num_cols + cat_cols]

# --- 3. Preprocesamiento ---
preprocessor = ColumnTransformer(transformers=[
    ('num', 'passthrough', num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

# --- 4. Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- 5. Crear pipeline con XGBoost ---
xgb_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
])

# --- 6. Entrenar modelo ---
xgb_model.fit(X_train, y_train)

# --- 7. Guardar modelo entrenado ---
with open("final_model.pkl", "wb") as f:
    pickle.dump(xgb_model, f)

print("✅ Modelo entrenado y guardado como 'final_model.pkl'")
