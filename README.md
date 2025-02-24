# crypto-ml-predictor

## Descripción

Este proyecto implementa un pipeline completo para trabajar con datos de criptomonedas y predecir señales de trading utilizando machine learning. Se utilizan diversos indicadores técnicos y se aplica un modelo XGBoost para generar predicciones de compra, venta o mantenimiento (hold).

## Tabla de Contenidos

* [Descripción](#descripcion)
* [Tabla de Contenidos](#tabla-de-contenidos)
* [Tecnologías Utilizadas](#tecnologias-utilizadas)
* [Instalación](#instalacion)
* [Uso](#uso)
* [Funcionalidades](#funcionalidades)
* [Evaluación del Modelo](#evaluacion-del-modelo)
* [Conclusiones](#conclusiones)
* [Autor](#autor)

## Tecnologías Utilizadas

* **Lenguaje de programación:** Python
* **Librerías:**
    * pandas
    * numpy
    * requests
    * scikit-learn
    * xgboost
    * matplotlib
    * optuna
    * joblib
    * imbalanced-learn (SMOTE)

## Instalación

1. Clona este repositorio: `git clone https://github.com/PaulinalA/crypto-ml-predictor.git`
2. Instala las dependencias: `pip install -r requirements.txt`
   (Puedes generar el archivo `requirements.txt` con `pip freeze > requirements.txt`)

## Uso

1. Abre el notebook `crypto-ml-predictor.ipynb` en Google Colab.
2. Sigue las instrucciones dentro del notebook para ejecutar el código y entrenar el modelo.

## Funcionalidades

* **Obtención de datos:** Se conecta a la API de LunarCrush para obtener datos históricos de criptomonedas.
* **Procesamiento de datos:** Limpieza, remuestreo y cálculo de indicadores técnicos (volatilidad, RSI, MACD, ATR, Bandas de Bollinger, EMA, Force Index, Williams %R, volatilidad relativa, tasa de cambio, acumulación/distribución, MFI y ADX).
* **Ingeniería de características:** Se generan nuevas características basadas en los indicadores técnicos.
* **Balanceo de clases:** Se aplica SMOTE para balancear el dataset de entrenamiento.
* **Entrenamiento del modelo:** Se entrena un clasificador XGBoost con parámetros predefinidos (optimizables con optuna).
* **Evaluación del modelo:** Se calculan métricas como precisión, ROC AUC, se genera un reporte de clasificación y se visualiza la matriz de confusión.
* **Visualización de la importancia de las características:** Se genera un gráfico que muestra la relevancia de cada característica utilizada en el modelo.

## Evaluación del Modelo

El modelo entrenado alcanzó los siguientes resultados en el conjunto de prueba:

* Precisión en Test: 0.74
* ROC AUC Score: 0.88

El reporte de clasificación muestra un desempeño consistente en las tres categorías (Hold, Buy y Sell), con f1-scores de aproximadamente 0.69 para la clase 0 y 0.77 para las clases 1 y 2.

## Conclusiones

Este proyecto demuestra la viabilidad de utilizar machine learning para predecir señales de trading en el mercado de criptomonedas. Los resultados obtenidos son prometedores y sugieren que el modelo tiene una buena capacidad para discriminar entre las diferentes señales de trading.

## Autor

Paulina Peralta
