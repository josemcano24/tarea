import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

# Título y descripción
st.title("🔍 Predicción con Regresión Lineal Simple")
st.write("Aplicación interactiva para entrenar un modelo de regresión lineal y visualizar las predicciones.")
st.write("Sube un archivo CSV y selecciona la variable dependiente (Y) y la variable independiente (X).")

# 1. Cargar datos
st.subheader("1️⃣ Cargar datos")
uploaded_file = st.file_uploader("Sube un archivo CSV con tus datos", type=["csv"])

if uploaded_file is not None:
    # Leer datos
    data = pd.read_csv(uploaded_file)
    st.write("Vista previa de los datos:")
    st.dataframe(data.head())

    # Seleccionar columnas
    columnas = data.columns.tolist()
    x_col = st.selectbox("Selecciona la variable independiente (X)", columnas)
    y_col = st.selectbox("Selecciona la variable dependiente (Y)", columnas)

    # Preparar datos para el modelo
    X = data[[x_col]]
    y = data[y_col]

    # --- INICIO DE CÓDIGO AÑADIDO ---

    # 2. Entrenamiento del modelo
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    st.success("Modelo entrenado correctamente")

    # 3. Mostrar ecuación y R²
    st.subheader("2️⃣ Ecuación del modelo y $R^2$")
    
    # Obtener coeficiente e intercepto
    coef = model.coef_[0]
    intercept = model.intercept_

    # Mostrar Ecuación
    st.write("**Ecuación del modelo:**")
    # Usamos st.latex para mostrar la fórmula matemática como en el PDF
    st.latex(f"Y = {coef:.2f}X + {intercept:.2f}")

    # Calcular y mostrar el R²
    r2 = r2_score(y, y_pred)
    
    # Usamos st.metric para un formato destacado como en el PDF
    st.metric(label="Coeficiente de Determinación ($R^2$)", value=f"{r2:.4f}")
    st.write(f"El valor de R² es: {r2:.4f}")

    # 4. Predicción interactiva
    st.subheader("3️⃣ Realiza una predicción")
    
    # Usamos st.number_input para que el usuario ingrese un valor
    default_val = float(X.mean()) # Usamos la media como valor por defecto
    new_x = st.number_input(f"Introduce un valor para {x_col}:", value=default_val)
    
    # Preparamos el valor para el modelo (debe ser un array 2D)
    new_x_reshaped = np.array([[new_x]])
    prediction = model.predict(new_x_reshaped)
    
    # Mostramos la predicción
    st.write(f"**Predicción para {x_col} = {new_x}: {prediction[0]:.2f}**")

    # 5. Generar gráfico
    st.subheader("4️⃣ Visualización del modelo")
    
    # Crear gráfico con Matplotlib
    fig, ax = plt.subplots()

    # 1. Datos reales (puntos azules)
    ax.scatter(X, y, color='blue', label='Datos reales')
    
    # 2. Línea de regresión (línea roja)
    ax.plot(X, y_pred, color='red', label='Línea de regresión')
    
    # 3. Predicción (punto verde grande)
    ax.scatter(new_x, prediction, color='green', s=100, label='Predicción', zorder=5)
    
    # Estilo y etiquetas (como en el PDF)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend()
    ax.grid(True)
    
    # Mostrar gráfico en Streamlit
    st.pyplot(fig)
    
    # --- FIN DE CÓDIGO AÑADIDO ---

else:
    st.info("☝️ Sube un archivo CSV para continuar.")
