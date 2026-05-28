import streamlit as st

st.title("Forecasting Ventas")
st.write("Bienvenido a la app de Streamlit para el proyecto de forecasting.")

st.sidebar.header("Opciones")
option = st.sidebar.selectbox("Selecciona una opción", ["Resumen", "Predicción"])

if option == "Resumen":
    st.write("Aquí irá el resumen del análisis de datos y resultados.")
else:
    st.write("Aquí irá la sección de predicción y visualización.")
