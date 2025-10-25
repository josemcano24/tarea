import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

# 1. Función para crear los datos de ejemplo por defecto
def get_default_data():
    """
    Crea un DataFrame con los datos de ejemplo vistos en el PDF de la tarea.
    (horas vs calificacion)
    """
    data = {
        'horas': [1.0, 2.0, 3.0, 4.0, 5.0],
        'calificacion': [50, 55, 60, 70, 80]
    }
    return pd.DataFrame(data)

# Título y descripción
st.title("🔍 Predicción con Regresión Lineal Simple")
st.write("Aplicación interactiva para entrenar un modelo de regresión lineal y visualizar las predicciones.")
st.write("Sube un archivo CSV y selecciona la variable dependiente (Y) y la variable independiente (X).")

# 1. Cargar datos
# --- CAMBIO AQUÍ: Se quitó el emoji "1️⃣" ---
st.subheader("1. Cargar datos") 
uploaded_file = st.file_uploader("Sube un archivo CSV (Opcional, se usará un ejemplo por defecto)", type=["csv"])

# Lógica para usar datos de ejemplo o los subidos
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Vista previa de los datos (cargados por el usuario):")
else:
    data = get_default_data()
    st.write("Mostrando datos de ejemplo por defecto:")

st.dataframe(data.head())

# Seleccionar columnas
columnas = data.columns.tolist()
default_x_index = 0
default_y_index = 1
if 'horas' in columnas:
    default_x_index = columnas.index('horas')
if 'calificacion' in columnas:
    default_y_index = columnas.index('calificacion')

x_col = st.selectbox("Selecciona la variable independiente (X)", columnas, index=default_x_index)
y_col = st.selectbox("Selecciona la variable dependiente (Y)", columnas, index=default_y_index)

# Preparar datos para el modelo
X = data[[x_col]]
y = data[y_col]

# 2. Entrenamiento del modelo
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

st.success("Modelo entrenado correctamente")

# 3. Mostrar ecuación y R²
# --- CAMBIO AQUÍ: Se quitó el emoji "2️⃣" ---
st.subheader("2. Ecuación del modelo y R²")
    
coef = model.coef_[0]
intercept = model.intercept_

st.write("**Ecuación del modelo:**")
st.latex(f"Y = {coef:.2f}X + {intercept:.2f}")

r2 = r2_score(y, y_pred)
    
st.metric(label="Coeficiente de Determinación ($R^2$)", value=f"{r2:.4f}")
st.write(f"El valor de R² es: {r2:.4f}")

# 4. Predicción interactiva
# --- CAMBIO AQUÍ: Se quitó el emoji "3️⃣" ---
st.subheader("3. Realiza una predicción")
    
default_val = float(X.mean()) 
new_x = st.number_input(f"Introduce un valor para {x_col}:", value=default_val)
    
new_x_reshaped = np.array([[new_x]])
prediction = model.predict(new_x_reshaped)
    
st.write(f"**Predicción para {x_col} = {new_x}: {prediction[0]:.2f}**")

# 5. Generar gráfico
# --- CAMBIO AQUÍ: Se quitó el emoji "4️⃣" ---
st.subheader("4. Visualización del modelo")
    
fig, ax = plt.subplots()
ax.scatter(X, y, color='blue', label='Datos reales')
ax.plot(X, y_pred, color='red', label='Línea de regresión')
ax.scatter(new_x, prediction, color='green', s=100, label='Predicción', zorder=5)
    
ax.set_xlabel(x_col)
ax.set_ylabel(y_col)
ax.legend()
ax.grid(True)
    
st.pyplot(fig)
