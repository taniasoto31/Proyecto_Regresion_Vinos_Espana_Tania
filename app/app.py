from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Configuración de la página
st.set_page_config(
    page_title="Predictor de Precios de Vinos de España",
    page_icon="🍷",
    layout="centered",
)

st.title("🍷 Predictor de Precios de Vinos de España")
st.write(
    "Esta aplicación utiliza un modelo de Machine Learning (Random Forest) entrenado "
    "para predecir el precio de los vinos españoles en función de sus características."
)

ARCHIVOS = ["modelo_final_vinos.pkl", "columnas_modelo.pkl", "categorias.pkl"]


@st.cache_resource
def cargar_artefactos():
    """Carga modelo, columnas de entrenamiento y categorías. Devuelve (artefactos, carpeta_buscada)."""
    base_dir = Path(__file__).resolve().parent

    # Carpetas posibles, en orden de prioridad
    for carpeta in [base_dir / "models", base_dir / "Models", base_dir,
                    base_dir.parent / "models", base_dir.parent / "Models"]:
        if all((carpeta / nombre).exists() for nombre in ARCHIVOS):
            modelo, columnas, categorias = (joblib.load(carpeta / nombre) for nombre in ARCHIVOS)
            # El modelo se entrenó con n_jobs=-1 (todos los núcleos). En la nube compartida eso puede
            # bloquear la predicción, así que se fuerza a 1 hilo (suficiente para una sola fila).
            try:
                modelo.set_params(n_jobs=1)
            except Exception:
                pass
            return (modelo, columnas, categorias), carpeta

    return None, base_dir / "models"


with st.spinner("Cargando modelo..."):
    artefactos, carpeta = cargar_artefactos()

if artefactos is None:
    st.error(f"❌ No se encontraron los archivos del modelo en: {carpeta}")
    existentes = [n for n in ARCHIVOS if (carpeta / n).exists()]
    faltantes = [n for n in ARCHIVOS if n not in existentes]
    st.write("✅ Encontrados:", existentes or "ninguno")
    st.write("❌ Faltan:", faltantes)
    st.info("💡 Copia los archivos que faltan a esa carpeta (y súbelos a GitHub si despliegas en la nube).")
    st.stop()

modelo, columnas, categorias = artefactos
st.success("¡Modelo cargado correctamente!")

st.markdown("---")

# Formulario de entradas para el usuario
st.subheader("📝 Introduce las características del vino:")

col1, col2 = st.columns(2)

with col1:
    puntuacion = st.slider("Puntuación (rating)", min_value=4.2, max_value=4.9, value=4.2, step=0.1)
    num_resenas = st.number_input("Número de reseñas", min_value=25, max_value=33000, value=400, step=10)
    cuerpo = st.slider("Cuerpo (escala 2–5)", min_value=2.0, max_value=5.0, value=4.0, step=1.0)

with col2:
    acidez = st.slider("Acidez (escala 1–3)", min_value=1.0, max_value=3.0, value=3.0, step=1.0)
    region = st.selectbox("Región", categorias["region"])
    tipo = st.selectbox("Tipo de vino", categorias["tipo"])

# Botón para realizar la predicción
if st.button("🚀 Calcular Precio Estimado", type="primary"):
    datos_entrada = pd.DataFrame([{
        "puntuacion": puntuacion,
        "num_reseñas": num_resenas,
        "cuerpo": cuerpo,
        "acidez": acidez,
        "region": region,
        "tipo": tipo,
    }])

    try:
        # Mismo preprocesado que en el entrenamiento: one-hot + alineación con las columnas del modelo
        datos_entrada = pd.get_dummies(datos_entrada, columns=["region", "tipo"])
        datos_entrada = datos_entrada.reindex(columns=columnas, fill_value=0)

        prediccion = modelo.predict(datos_entrada)[0]
        st.markdown("### 📊 Resultado de la Predicción:")
        st.metric(label="Precio Estimado del Vino", value=f"{prediccion:.2f} €")

        if prediccion > 150:
            st.warning(
                "⚠️ El modelo es fiable sobre todo para vinos de hasta ~150 € (≈ 93 % de los datos). "
                "Por encima de ese precio, la estimación es orientativa y suele subestimar el valor real."
            )
        st.caption("Error medio del modelo en datos de prueba (MAE): ≈ 33 €.")

    except Exception as e:
        st.error(f"Ocurrió un error al hacer la predicción: {e}")
        st.info("💡 Comprueba que 'columnas_modelo.pkl' corresponde al modelo cargado.")
