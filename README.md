# Crypto-ML-Predictor

## Descripción
Este proyecto implementa un sistema avanzado de trading algorítmico para criptomonedas que utiliza machine learning para predecir señales de compra, venta o mantenimiento (hold). El sistema analiza datos históricos de Bitcoin, calcula más de 15 indicadores técnicos y entrena un modelo XGBoost optimizado para generar predicciones de alta precisión.

![Bitcoin Trading Signals](prediction_signals.png)

## Tabla de Contenidos
* [Descripción](#descripción)
* [Características Principales](#características-principales)
* [Tecnologías Utilizadas](#tecnologías-utilizadas)
* [Arquitectura del Sistema](#arquitectura-del-sistema)
* [Instalación](#instalación)
* [Uso](#uso)
* [Pipeline de Datos](#pipeline-de-datos)
* [Indicadores Técnicos](#indicadores-técnicos)
* [Modelo Predictivo](#modelo-predictivo)
* [Evaluación del Modelo](#evaluación-del-modelo)
* [Resultados](#resultados)
* [Limitaciones y Trabajo Futuro](#limitaciones-y-trabajo-futuro)
* [Licencia](#licencia)
* [Autor](#autor)

## Características Principales
- **Análisis Técnico Avanzado**: Implementación de más de 15 indicadores técnicos para capturar diferentes aspectos del mercado.
- **Predicción Multi-clase**: Clasificación de señales en tres categorías (Comprar, Vender, Mantener).
- **Optimización de Hiperparámetros**: Búsqueda automatizada de los mejores parámetros para el modelo.
- **Balanceo de Clases**: Tratamiento de datos desbalanceados para mejorar el rendimiento predictivo.
- **Visualización de Resultados**: Gráficos interpretables de señales de trading e importancia de características.
- **Manejo de Fallas en API**: Sistema de respaldo para obtener datos alternativos cuando la API principal no está disponible.

## Tecnologías Utilizadas
* **Lenguaje de programación:** Python 3.8+
* **Análisis de datos y manipulación:**
  * pandas 1.3.0+
  * numpy 1.20.0+
* **Machine Learning:**
  * scikit-learn 1.0.0+
  * xgboost 1.5.0+
  * imbalanced-learn 0.8.0+ (SMOTE)
  * optuna 2.10.0+ (optimización de hiperparámetros)
* **Visualización:**
  * matplotlib 3.4.0+
* **Otras librerías:**
  * requests 2.26.0+
  * joblib 1.1.0+

## Arquitectura del Sistema
El sistema está diseñado en un pipeline modular de 5 etapas:

1. **Adquisición de Datos**: Conexión a APIs (LunarCrush o CryptoCompare) con manejo de fallos.
2. **Procesamiento de Datos**: Limpieza, cálculo de indicadores y generación de características.
3. **Entrenamiento del Modelo**: XGBoost optimizado con balanceo SMOTE.
4. **Evaluación y Análisis**: Métricas de rendimiento y análisis de importancia de características.
5. **Predicción y Visualización**: Generación de señales de trading con niveles de confianza.

## Instalación

### Requisitos
- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- Git

### Pasos
1. Clona este repositorio:
   ```bash
   git clone https://github.com/PaulinalA/crypto-ml-predictor.git
   cd crypto-ml-predictor
   ```

2. Crea un entorno virtual (opcional pero recomendado):
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```

3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

### Configuración de API (opcional)
Para utilizar la API de LunarCrush, registra una cuenta en [lunarcrush.com](https://lunarcrush.com) y obtén una API key. Luego, configúrala en el script (o utiliza un archivo .env).

## Uso

### Modo Básico
```python
# Importar la función principal
from crypto_predictor import main

# Ejecutar el pipeline completo
main()
```

### Modo Avanzado
```python
# Importar las funciones necesarias
from crypto_predictor import fetch_crypto_data, process_data, train_model, predict_and_visualize

# Obtener datos
df = fetch_crypto_data(coin_id='1', api_key='TU_API_KEY')

# Procesar datos
processed_df = process_data(df, freq='D')  # Frecuencia diaria

# Definir características y target
features = ['open', 'high', 'low', 'market_cap', 'rsi', 'macd', 'atr_volatility', 
            'bollinger_upper', 'bollinger_lower', 'williams_r', 'relative_volatility', 
            'rate_of_change', 'accumulation_distribution', 'mfi', 'adx']
X = processed_df[features]
y = processed_df['target']

# Entrenar modelo con optimización de hiperparámetros
model, scaler = train_model(X, y, optimize=True)

# Generar predicciones y visualizar resultados
results = predict_and_visualize(model, scaler, processed_df, features)

# Mostrar últimas predicciones
print(results[['close', 'signal', 'prediction_proba']].tail(10))
```

## Pipeline de Datos

### Adquisición de Datos
- **Fuente primaria**: API LunarCrush (requiere autenticación)
- **Fuente secundaria**: API CryptoCompare (sin autenticación)
- **Generación sintética**: Como último recurso si ambas APIs fallan

### Preprocesamiento
- Limpieza de valores nulos
- Remuestreo temporal (diario/horario)
- Normalización de características (StandardScaler)
- Generación de variable objetivo basada en cambios porcentuales:
  - **Comprar (1)**: Cambio > +2%
  - **Vender (2)**: Cambio < -2%
  - **Mantener (0)**: Cambio entre -2% y +2%

## Indicadores Técnicos
Se calculan múltiples indicadores técnicos para capturar diferentes aspectos del mercado:

| Indicador | Descripción | Importancia |
|-----------|-------------|-------------|
| RSI | Índice de Fuerza Relativa (condiciones de sobrecompra/sobreventa) | Alta |
| MACD | Convergencia/Divergencia de Medias Móviles | Alta |
| ATR | Rango Verdadero Promedio (volatilidad) | Media |
| Bandas de Bollinger | Canales de volatilidad | Alta |
| Williams %R | Posición relativa del precio | Media |
| EMA | Medias Móviles Exponenciales (50 y 200 períodos) | Alta |
| Acumulación/Distribución | Flujo de dinero | Baja |
| MFI | Índice de Flujo de Dinero | Media |
| ADX | Índice Direccional Promedio (fuerza de tendencia) | Alta |
| Volatilidad Histórica | Basada en desviación estándar | Media |
| Tasa de Cambio | Impulso del precio | Media |

## Modelo Predictivo

### Algoritmo
XGBoost (Extreme Gradient Boosting) - Clasificador multiclase

### Hiperparámetros Optimizados
- n_estimators: 500-1000
- max_depth: 8-14
- learning_rate: 0.01-0.02
- subsample: 0.8-0.9
- colsample_bytree: 0.5-0.6
- min_child_weight: 1-3

### Manejo del Desbalance de Clases
SMOTE (Synthetic Minority Over-sampling Technique) para equilibrar la distribución de clases

## Evaluación del Modelo

### Métricas de Rendimiento
- **Precisión**: ~0.74
- **ROC AUC**: ~0.88 (multiclase one-vs-rest)
- **F1-Score por clase**:
  - Mantener (0): ~0.69
  - Comprar (1): ~0.77
  - Vender (2): ~0.77

### Matriz de Confusión
Representación visual del rendimiento del modelo por clase:
![Matriz de Confusión](confusion_matrix.png)

### Importancia de Características
Análisis de las características más influyentes en las predicciones:
![Importancia de Características](feature_importance.png)

## Resultados
El modelo demuestra capacidad para identificar oportunidades de compra y venta en el mercado de Bitcoin con una precisión significativamente superior al azar (74% vs 33% esperado para 3 clases equilibradas).

Las características más importantes para las predicciones resultaron ser:
1. RSI (Índice de Fuerza Relativa)
2. Diferencia MACD
3. Volatilidad relativa
4. ATR (Rango Verdadero Promedio)
5. MFI (Índice de Flujo de Dinero)

## Limitaciones y Trabajo Futuro

### Limitaciones Actuales
- Dependencia de APIs externas para datos
- No considera eventos fundamentales o sentimiento de mercado
- Optimizado principalmente para Bitcoin (BTC)

### Trabajo Futuro
- Incorporar análisis de sentimiento desde redes sociales
- Implementar backtesting completo con simulación de operaciones
- Expandir a múltiples criptomonedas
- Desarrollar una interfaz web para visualización en tiempo real
- Implementar aprendizaje por refuerzo para optimización de estrategias

## Licencia
Este proyecto está licenciado bajo la Licencia MIT - consulte el archivo [LICENSE](LICENSE) para más detalles.

## Autor
Paulina Peralta

