import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

# --- CÓDIGO NUEVO ---
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
st.subheader("1️⃣ Cargar datos") [cite: 36]
# --- TEXTO MODIFICADO ---
uploaded_file = st.file_uploader("Sube un archivo CSV (Opcional, se usará un ejemplo por defecto)", type=["csv"]) [cite: 37]

# --- LÓGICA MODIFICADA ---
# Ahora, 'data' existirá siempre, ya sea la cargada o la de por defecto.
if uploaded_file is not None:
    # Si el usuario sube un archivo, úsalo
    data = pd.read_csv(uploaded_file)
    st.write("Vista previa de los datos (cargados por el usuario):")
else:
    # Si no, usa los datos por defecto
    data = get_default_data()
    st.write("Mostrando datos de ejemplo por defecto:")

# --- TODO EL CÓDIGO DE AQUÍ ABAJO AHORA ESTÁ FUERA DEL 'IF' ---

st.dataframe(data.head()) # [cite: 41]

# Seleccionar columnas
columnas = data.columns.tolist()

# --- LÓGICA MEJORADA ---
# Intentar pre-seleccionar 'horas' y 'calificacion' si existen (para el ejemplo)
default_x_index = 0
default_y_index = 1 # Por defecto son la columna 0 y 1

if 'horas' in columnas:
    default_x_index = columnas.index('horas')
if 'calificacion' in columnas:
    default_y_index = columnas.index('calificacion')
# --- FIN LÓGICA MEJORADA ---

x_col = st.selectbox("Selecciona la variable independiente (X)", columnas, index=default_x_index) [cite: 46, 47]
y_col = st.selectbox("Selecciona la variable dependiente (Y)", columnas, index=default_y_index) [cite: 48, 49]

# Preparar datos para el modelo
X = data[[x_col]]
y = data[y_col]

# 2. Entrenamiento del modelo
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

st.success("Modelo entrenado correctamente") [cite: 50]

# 3. Mostrar ecuación y R²
st.subheader("2️⃣ Ecuación del modelo y $R^2$")
    
# Obtener coeficiente e intercepto
coef = model.coef_[0]
intercept = model.intercept_

# Mostrar Ecuación
st.write("**Ecuación del modelo:**") [cite: 67]
# Usamos st.latex para mostrar la fórmula matemática como en el PDF
st.latex(f"Y = {coef:.2f}X + {intercept:.2f}") # [cite: 73]

# Calcular y mostrar el R²
r2 = r2_score(y, y_pred)
    
# Usamos st.metric para un formato destacado como en el PDF
st.metric(label="Coeficiente de Determinación ($R^2$)", value=f"{r2:.4f}") [cite: 69, 70]
st.write(f"El valor de R² es: {r2:.4f}") [cite: 71]

# 4. Predicción interactiva
st.subheader("3️⃣ Realiza una predicción") [cite: 75]
    
# Usamos st.number_input para que el usuario ingrese un valor
# Usamos la media como valor por defecto, como en el ejemplo (3.00)
default_val = float(X.mean()) 
new_x = st.number_input(f"Introduce un valor para {x_col}:", value=default_val) [cite: 76, 77]
    
# Preparamos el valor para el modelo (debe ser un array 2D)
new_x_reshaped = np.array([[new_x]])
prediction = model.predict(new_x_reshaped)
    
# Mostramos la predicción
st.write(f"**Predicción para {x_col} = {new_x}: {prediction[0]:.2f}**") [cite: 78]

# 5. Generar gráfico
st.subheader("4️⃣ Visualización del modelo") [cite: 79]
    
# Crear gráfico con Matplotlib
fig, ax = plt.subplots()

# 1. Datos reales (puntos azules)
ax.scatter(X, y, color='blue', label='Datos reales') [cite: 82]
    
# 2. Línea de regresión (línea roja)
ax.plot(X, y_pred, color='red', label='Línea de regresión') [cite: 83]
    
# 3. Predicción (punto verde grande)
ax.scatter(new_x, prediction, color='green', s=100, label='Predicción', zorder=5) [cite: 83]
    
# Estilo y etiquetas (como en el PDF)
ax.set_xlabel(x_col) [cite: 92]
ax.set_ylabel(y_col) [cite: 80]
ax.legend()
ax.grid(True)
    
# Mostrar gráfico en Streamlit
st.pyplot(fig)
