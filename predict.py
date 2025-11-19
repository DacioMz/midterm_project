# predict.py
# ---------------------------
# Script to load the trained model and make predictions
# ---------------------------

import pandas as pd
import pickle

# --- 1. Cargar modelo entrenado ---
with open("final_model.pkl", "rb") as f:
    model = pickle.load(f)

# --- 2. Crear datos de ejemplo para predecir ---
# Asegurate de incluir todas las columnas usadas por el modelo
new_data = pd.DataFrame({
    "price": [100, 200],
    "initial_quantity": [10, 5],
    "sold_quantity": [2, 1],
    "condition": ["new", "used"],  # si tu modelo ya codifica la variable target, esta puede no ser necesaria
    "buying_mode": ["buy_it_now", "auction"],
    "listing_type_id": ["gold_special", "gold_pro"],
})

# --- 3. Hacer predicciones ---
preds = model.predict(new_data)
pred_probs = model.predict_proba(new_data)  # opcional: obtener probabilidades

# --- 4. Mostrar resultados ---
print("Predictions:", preds)
print("Prediction probabilities:\n", pred_probs)
