import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

# Paleta de colores
PRIMARY_COLOR = '#667eea'
SECONDARY_COLOR = '#764ba2'
BLACK_FRIDAY_DAY = 28

st.set_page_config(page_title="Simulación de Ventas Noviembre 2025", layout="wide", page_icon="📈")

# --- Sidebar ---
st.sidebar.title("🎛️ Controles de Simulación")

# Cargar datos y modelo con manejo de errores
@st.cache_data

def load_data():
    try:
        df = pd.read_csv("../data/processed/inferencia_df_transformado.csv")
    except Exception:
        df = pd.read_csv("data/processed/inferencia_df_transformado.csv")
    return df

@st.cache_resource

def load_model():
    try:
        model = joblib.load("../models/modelo_final.joblib")
    except Exception:
        model = joblib.load("models/modelo_final.joblib")
    return model

df = load_data()
model = load_model()
model.feature_names_in_



# --- Controles del sidebar ---
productos = df['nombre'].unique().tolist()
producto_sel = st.sidebar.selectbox("Selecciona producto", productos)
descuento = st.sidebar.slider("Ajuste de descuento (%)", -50, 50, 0, 5)
escenarios = {"Actual (0%)": 0, "Competencia -5%": -5, "Competencia +5%": 5}
esc_sel = st.sidebar.radio("Escenario de competencia", list(escenarios.keys()))
run_sim = st.sidebar.button("🚀 Simular Ventas", use_container_width=True)

# --- Función de predicción recursiva ---
def simular_predicciones(df_prod, model, descuento, ajuste_comp):
    df_pred = df_prod.copy().sort_values('fecha').reset_index(drop=True)
    st.write(df_pred.columns.tolist())
    dias = df_pred.shape[0]
    predicciones = []
    lags_cols = [f'unidades_vendidas_lag{i}' for i in range(1,8)]
    ma7_col = 'unidades_vendidas_ma7'
    # Inicializar lags y ma7 con los valores del archivo para el día 1
    for i in range(dias):
        # --- Actualizar precios y descuentos ---
        df_pred.at[i, 'precio_venta'] = df_pred.at[i, 'precio_base'] * (1 + descuento/100)
        ##Before it was doing:
        for col in ['Amazon', 'Decathlon', 'Deporvillage']:
    
            # Crear columna si no existe
            if col not in df_pred.columns:
                df_pred[col] = df_pred['precio_competencia']

            # Ajustar precio competencia
            df_pred.at[i, col] = df_pred.at[i, col] * (1 + ajuste_comp/100)
        #for col in ['Amazon', 'Decathlon', 'Deporvillage']:
            #if col not in df_pred.columns:
                #df_pred[col] = df_pred['precio_competencia']
        df_pred.at[i, 'precio_competencia'] = df_pred.loc[i, ['Amazon', 'Decathlon', 'Deporvillage']].mean()
        df_pred.at[i, 'descuento_porcentaje'] = (df_pred.at[i, 'precio_venta'] - df_pred.at[i, 'precio_base']) / df_pred.at[i, 'precio_base'] * 100
        df_pred.at[i, 'ratio_precio'] = df_pred.at[i, 'precio_venta'] / df_pred.at[i, 'precio_competencia']
        # --- Seleccionar features ---
        X = df_pred.loc[[i], model.feature_names_in_]
        # --- Predecir ---
        pred = model.predict(X)[0]
        predicciones.append(pred)
        # --- Actualizar lags y ma7 para el siguiente día ---
        if i+1 < dias:
            # Desplazar lags
            for lag in range(7,1,-1):
                df_pred.at[i+1, f'unidades_vendidas_lag{lag}'] = df_pred.at[i, f'unidades_vendidas_lag{lag-1}']
            df_pred.at[i+1, 'unidades_vendidas_lag1'] = pred
            # Actualizar ma7
            ultimos_7 = predicciones[max(0,i-5):i+1]  # incluye el día actual y hasta 6 previos
            if len(ultimos_7) < 7:
                # Rellenar con los valores originales del archivo si faltan días
                faltan = 7 - len(ultimos_7)
                orig = [df_pred.at[i+1, f'unidades_vendidas_lag{j}'] for j in range(faltan,0,-1)]
                ultimos_7 = orig + ultimos_7
            df_pred.at[i+1, ma7_col] = np.mean(ultimos_7)
    df_pred['unidades_predichas'] = np.round(predicciones,0)
    df_pred['ingresos_predichos'] = df_pred['unidades_predichas'] * df_pred['precio_venta']
    return df_pred

# --- Simulación y visualización ---
def mostrar_dashboard(df, producto_sel, descuento, esc_sel):
    st.title(f"📊 Simulación de Ventas Noviembre 2025 - {producto_sel}")
    st.markdown("---")
    df_prod = df[df['nombre'] == producto_sel].copy().sort_values('fecha')
    ajuste_comp = escenarios[esc_sel]
    with st.spinner('Calculando predicciones recursivas...'):
        df_pred = simular_predicciones(df_prod, model, descuento, ajuste_comp)
    # --- KPIs ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Unidades totales", f"{int(df_pred['unidades_predichas'].sum()):,}")
    col2.metric("Ingresos totales (€)", f"{df_pred['ingresos_predichos'].sum():,.2f}")
    col3.metric("Precio medio venta (€)", f"{df_pred['precio_venta'].mean():.2f}")
    col4.metric("Descuento medio (%)", f"{df_pred['descuento_porcentaje'].mean():.2f}")
    st.markdown("---")
    # --- Gráfico de predicción diaria ---
    fig, ax = plt.subplots(figsize=(12,5))
    sns.lineplot(x=pd.to_datetime(df_pred['fecha']).dt.day, y=df_pred['unidades_predichas'], marker='o', color=PRIMARY_COLOR, ax=ax)
    ax.axvline(BLACK_FRIDAY_DAY, color='red', linestyle='--', lw=2, label='Black Friday')
    bf_idx = df_pred[df_pred['dia_mes'] == BLACK_FRIDAY_DAY].index[0]
    ax.scatter(BLACK_FRIDAY_DAY, df_pred.loc[bf_idx, 'unidades_predichas'], color='red', s=100, zorder=5)
    ax.annotate('Black Friday', xy=(BLACK_FRIDAY_DAY, df_pred.loc[bf_idx, 'unidades_predichas']), xytext=(BLACK_FRIDAY_DAY+1, df_pred['unidades_predichas'].max()*0.9),
                arrowprops=dict(facecolor='red', shrink=0.05), fontsize=12, color='red', weight='bold')
    ax.set_xlabel('Día de noviembre')
    ax.set_ylabel('Unidades predichas')
    ax.set_title('Predicción diaria de unidades vendidas')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    st.markdown("---")
    # --- Tabla detallada ---
    tabla = df_pred[['fecha','nombre','dia_semana_es','precio_venta','precio_competencia','descuento_porcentaje','unidades_predichas','ingresos_predichos']].copy()
    tabla = tabla.rename(columns={
        'fecha':'Fecha',
        'nombre':'Producto',
        'dia_semana_es':'Día semana',
        'precio_venta':'Precio venta (€)',
        'precio_competencia':'Precio competencia (€)',
        'descuento_porcentaje':'Descuento (%)',
        'unidades_predichas':'Unidades predichas',
        'ingresos_predichos':'Ingresos (€)'
    })
    tabla['Fecha'] = pd.to_datetime(tabla['Fecha']).dt.strftime('%d/%m/%Y')
    tabla['Unidades predichas'] = tabla['Unidades predichas'].astype(int)
    tabla['Ingresos (€)'] = tabla['Ingresos (€)'].map(lambda x: f"{x:,.2f} €")
    tabla['Precio venta (€)'] = tabla['Precio venta (€)'].map(lambda x: f"{x:,.2f}")
    tabla['Precio competencia (€)'] = tabla['Precio competencia (€)'].map(lambda x: f"{x:,.2f}")
    tabla['Descuento (%)'] = tabla['Descuento (%)'].map(lambda x: f"{x:.2f}")
    # Destacar Black Friday
    tabla['__color__'] = ''
    tabla.loc[tabla['Fecha'].str.startswith(f'{BLACK_FRIDAY_DAY:02d}/'), '__color__'] = 'background-color: #ffe6e6; font-weight: bold;'
    tabla.loc[tabla['Fecha'].str.startswith(f'{BLACK_FRIDAY_DAY:02d}/'), 'Día semana'] += ' 🛍️'
    st.dataframe(tabla.drop(columns='__color__'), use_container_width=True, hide_index=True)
    st.markdown("---")
    # --- Comparativa de escenarios ---
    st.subheader('Comparativa de escenarios de competencia')
    colA, colB, colC = st.columns(3)
    for i, (esc, ajuste) in enumerate(escenarios.items()):
        with [colA, colB, colC][i]:
            df_esc = simular_predicciones(df_prod, model, descuento, ajuste)
            st.metric(f"{esc}", f"{int(df_esc['unidades_predichas'].sum()):,} uds", f"{df_esc['ingresos_predichos'].sum():,.2f} €")
    st.info('Puedes ajustar el descuento y el escenario de competencia para simular diferentes resultados. Pulsa "Simular Ventas" para actualizar.')

if run_sim:
    mostrar_dashboard(df, producto_sel, descuento, esc_sel)
else:
    st.title("📊 Simulación de Ventas Noviembre 2025")
    st.info('Selecciona un producto y configura los controles en la barra lateral. Pulsa "Simular Ventas" para ver la predicción.')
