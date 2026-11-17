# 🍷 Predicción de Precios de Vino Español

Proyecto de regresión con *Machine Learning* para estimar el **precio (€)** de un vino español a partir de sus características. Incluye análisis exploratorio, comparación de modelos, informe ejecutivo y una **aplicación web interactiva** desplegada con Streamlit.

**🔗 Aplicación:** [Abrir la app](https://proyectoregresionvinosespanatania-afkyeygsiqmybcoydvxxmg.streamlit.app/)  
**💻 Código:** [Repositorio en GitHub](https://github.com/Bootcamp-DA-P3/Proyecto_Regresion_Vinos_Espana_Tania)

---

## 📌 Resumen

- **Dataset:** [Spanish Wine Quality Dataset](https://www.kaggle.com/datasets/fedesoriano/spanish-wine-quality-dataset) (Kaggle). Tras eliminar nulos en las variables clave quedan 6.331 vinos (se pierde ≈ 16 % de las filas).
- **Objetivo:** predecir `precio` a partir de `puntuacion`, `num_reseñas`, `cuerpo`, `acidez`, `region` y `tipo`.
- **Modelo final:** Random Forest Regressor optimizado con GridSearchCV (5 folds).
- **Dashboard:** panel en Power BI incluido en Dashboard/Vinos_Espana.pbix.
- **Hiperparámetros óptimos:** `n_estimators=150`, `max_depth=10`, `max_features=0.5`, `min_samples_leaf=10`, `min_samples_split=15`.

## 📊 Resultados del modelo final

| Métrica | Train | Test | Brecha / Variación |
| :--- | :---: | :---: | :---: |
| R² | 0.5235 | 0.5059 | 3.36 % |
| RMSE | 112.20 € | 114.46 € | +2.01 % |
| MAE | 27.28 € | 33.22 € | +21.75 % |

### Comparativa de modelos

| Modelo | R² Test | Brecha R² | RMSE Test | MAE Test |
| :--- | :---: | :---: | :---: | :---: |
| Regresión Lineal (base) | 0.3712 | 4.02 % | 129.13 € | 50.12 € |
| Random Forest base | 0.4286 | 52.24 % | 123.10 € | 34.05 € |
| Random Forest regularizado | 0.5017 | 0.16 % | 114.95 € | 37.43 € |
| **Random Forest GridSearchCV** | **0.5059** | 3.36 % | 114.46 € | 33.22 € |

### Importancia de variables

| Variable | Importancia |
| :--- | :---: |
| Puntuación (`puntuacion`) | 62.4 % |
| Nº de reseñas (`num_reseñas`) | 23.5 % |
| Región (`region`) | 7.1 % |
| Tipo de vino (`tipo`) | 5.4 % |
| Cuerpo (`cuerpo`) | 1.5 % |
| Acidez (`acidez`) | 0.1 % |

### Error por tramo de precio (Test)

| Segmento | Nº vinos | Residuo medio | MAE |
| :--- | :---: | :---: | :---: |
| ≤ 150 € | 1.180 | −12.83 € | 17.52 € |
| 150 – 500 € | 62 | +11.87 € | 140.27 € |
| > 500 € | 25 | +492.61 € | 508.89 € |

> Residuo = Real − Predicción. Un residuo positivo indica que el modelo **subestima** el precio.

## ⚠️ Limitaciones

- El modelo es fiable sobre todo para vinos de **hasta ~150 €** (≈ 93 % de los datos de prueba).
- En vinos de más de 500 € **subestima de forma sistemática**; para ese rango la estimación es solo orientativa.
- El MAE (≈ 33 €) es comparable al precio mediano del dataset (29.15 €): el error típico es relevante para vinos de gama media y baja.
- `puntuacion` apenas varía en el dataset (la mayoría de vinos tiene 4.2), lo que limita el rendimiento.
- R² ≈ 0.51: el modelo explica aproximadamente la mitad de la variabilidad del precio.

## 🗂️ Estructura del repositorio

```
├── Dashboard/
│   └── Vinos_Espana.pbix           # Dashboard en Power BI
├── app/
│   └── app.py                      # Aplicación Streamlit
├── data/
│   └── wines_SPA.csv               # Dataset
├── informes/
│   └── Informe_Ejecutivo.ipynb     # Informe de rendimiento del modelo
├── models/
│   ├── modelo_final_vinos.pkl      # Modelo entrenado
│   ├── columnas_modelo.pkl         # Columnas de entrenamiento (para replicar get_dummies)
│   └── categorias.pkl              # Regiones y tipos válidos para la app
├── notebooks/
│   └── 01_eda_y_modelado_v2.ipynb  # EDA, modelado, evaluación y exportación
├── requirements.txt                # Dependencias (en la raíz, para Streamlit Cloud)
└── README.md
```

## 🚀 Ejecutar en local

```bash
git clone https://github.com/Bootcamp-DA-P3/Proyecto_Regresion_Vinos_Espana_Tania.git
cd Proyecto_Regresion_Vinos_Espana_Tania
pip install -r requirements.txt
streamlit run app/app.py
```

## ☁️ Despliegue en Streamlit Cloud

- **Main file path:** `app/app.py`
- **Python:** 3.12 (se elige al crear la app, en *Advanced settings*).
- `requirements.txt` en la raíz, con las **mismas versiones** que se usaron para entrenar. En especial `scikit-learn`, ya que el `.pkl` debe cargarse con la versión que lo generó.

## 🔁 Reentrenar el modelo

1. Ejecuta `01_eda_y_modelado_v2.ipynb` completo (por ejemplo, en Google Colab).
2. Descarga `modelo_final_vinos.pkl`, `columnas_modelo.pkl` y `categorias.pkl` con la celda de la sección 8.
3. Sustituye los tres archivos en `models/` y haz *commit* y *push*.
4. Si cambió la versión de scikit-learn, actualiza `requirements.txt`.

## 🔮 Próximos pasos

- Evaluar la transformación logarítmica del precio, comprobando su efecto por tramo de precio.
- Incorporar variables de reputación: interacción `bodega` × `region` o *target encoding* de `bodega`.
- Agrupar categorías poco frecuentes de `region` y `tipo` en "Otras".
- Mostrar un rango de confianza en la app (dispersión entre los árboles del bosque).

## 🛠️ Tecnologías

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · joblib · Streamlit · Power BI   

## 👩‍💻 Autora

Tania Soto — Noviembre 2026
