# midterm_project.py
import pandas as pd

# Cargar datos
file_path = "MLA_100K.jsonlines"
df = pd.read_json(file_path, lines=True, encoding='utf-8', nrows=50)

# Mostrar info básica
print("Información general del dataset:")
print(df.info())

# Columnas
print("\nColumnas:")
print(df.columns)

# Primeras filas
print("\nPrimeras filas:")
print(df.head())

# Estadísticas descriptivas de columnas numéricas
print("\nEstadísticas descriptivas:")
print(df.describe())

# Revisión de variables categóricas importantes
print("\nConteo por 'condition':")
print(df['condition'].value_counts())

print("\nConteo por 'buying_mode':")
print(df['buying_mode'].value_counts())

# Ejemplo: revisar precios
print("\nPrecios mínimos y máximos:")
print(df['price'].min(), df['price'].max())

# Si quieres graficar (opcional)
import matplotlib.pyplot as plt

plt.hist(df['price'], bins=10)
plt.title('Distribución de precios')
plt.xlabel('Precio')
plt.ylabel('Cantidad')
plt.show()


#### SEGUNDA PARTE 

import matplotlib.pyplot as plt
import seaborn as sns

# Histograma
sns.histplot(df['base_price'], bins=20)
plt.show()

# Boxplot
sns.boxplot(x=df['sold_quantity'])
plt.show()

# ===== EDA Completo =====
import matplotlib.pyplot as plt
import seaborn as sns

print("\n=== Exploratory Data Analysis (EDA) ===\n")

# 1️⃣ Distribuciones de variables numéricas
numericas = ['base_price', 'sold_quantity', 'available_quantity', 'initial_quantity']
for col in numericas:
    plt.figure(figsize=(8,4))
    sns.histplot(df[col], bins=20, kde=True)
    plt.title(f'Distribución de {col}')
    plt.show()
    
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot de {col}')
    plt.show()

# 2️⃣ Conteo de variables categóricas
categoricas = ['condition', 'buying_mode']
for col in categoricas:
    plt.figure(figsize=(6,4))
    sns.countplot(x=col, data=df)
    plt.title(f'Conteo de {col}')
    plt.show()

# 3️⃣ Relación entre variables numéricas y categóricas
for col in categoricas:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=col, y='base_price', data=df)
    plt.title(f'Precio vs {col}')
    plt.show()

# Relación entre variables numéricas
plt.figure(figsize=(8,6))
sns.scatterplot(x='base_price', y='sold_quantity', data=df)
plt.title('Sold Quantity vs Base Price')
plt.show()

# 4️⃣ Correlaciones
plt.figure(figsize=(8,6))
corr = df[numericas].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Matriz de Correlaciones')
plt.show()

# 5️⃣ Información adicional útil
print("\nValores únicos por variable categórica:")
for col in categoricas:
    print(f"{col}: {df[col].unique()}")
    
print("\nValores mínimos y máximos de las variables numéricas:")
for col in numericas:
    print(f"{col}: min={df[col].min()}, max={df[col].max()}")

## Quitamos los OUTLIERS

Q1 = df['base_price'].quantile(0.25)
Q3 = df['base_price'].quantile(0.75)
IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

# Filtramos los datos
df_sin_outliers = df[(df['base_price'] >= lower_limit) & (df['base_price'] <= upper_limit)]


### tercera parte

import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.boxplot(df['base_price'])
plt.title("Antes de quitar outliers")

plt.subplot(1,2,2)
plt.boxplot(df_sin_outliers['base_price'])
plt.title("Después de quitar outliers")

plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# -----------------------------
# 1️⃣ Información general
# -----------------------------
print("===== Dimensiones del dataset =====")
print(df_sin_outliers.shape)

print("\n===== Tipos de variables y nulos =====")
print(df_sin_outliers.info())

print("\n===== Primeras filas =====")
print(df_sin_outliers.head())

# -----------------------------
# 2️⃣ Estadísticas descriptivas
# -----------------------------
print("\n===== Estadísticas numéricas =====")
print(df_sin_outliers.describe())

categorical_cols = df_sin_outliers.select_dtypes(include='object').columns
print("\n===== Estadísticas de variables categóricas =====")
for col in categorical_cols:
    print(f"\nValores únicos y frecuencias en {col}:")
    print(df_sin_outliers[col].value_counts())

# -----------------------------
# 3️⃣ Distribución de variables numéricas
# -----------------------------
numeric_cols = df_sin_outliers.select_dtypes(include=np.number).columns
for col in numeric_cols:
    if df_sin_outliers[col].dropna().empty:
        continue  # saltar si no hay datos
    
    plt.figure(figsize=(8,4))
    sns.histplot(df_sin_outliers[col], bins=50, kde=True)
    plt.title(f'Distribución de {col}')
    plt.show()
    
    plt.figure(figsize=(8,4))
    sns.boxplot(x=df_sin_outliers[col].dropna())
    plt.title(f'Boxplot de {col}')
    plt.show()

# -----------------------------
# 4️⃣ Distribución de variables categóricas
# -----------------------------
for col in categorical_cols:
    plt.figure(figsize=(8,4))
    sns.countplot(y=col, data=df_sin_outliers, order=df_sin_outliers[col].value_counts().index)
    plt.title(f'Frecuencia de {col}')
    plt.show()

# -----------------------------
# 5️⃣ Correlaciones entre variables numéricas
# -----------------------------
plt.figure(figsize=(10,8))
corr = df_sin_outliers.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de correlación')
plt.show()

# -----------------------------
# 6️⃣ Boxplots: precio según categorías
# -----------------------------
for col in categorical_cols:
    if df_sin_outliers[col].dropna().empty:
        continue
    
    plt.figure(figsize=(10,5))
    sns.boxplot(x=col, y='base_price', data=df_sin_outliers.dropna(subset=[col, 'base_price']))
    plt.title(f'Base price por {col}')
    plt.xticks(rotation=45)
    plt.show()

# -----------------------------
# 7️⃣ Scatterplots: precio vs otras variables numéricas
# -----------------------------
for col in numeric_cols:
    if col != 'base_price':
        plt.figure(figsize=(8,4))
        sns.scatterplot(x=col, y='base_price', data=df_sin_outliers)
        plt.title(f'Base price vs {col}')
        plt.show()
