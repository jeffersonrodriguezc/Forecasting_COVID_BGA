# Forecasting_COVID_BGA

<img src="https://github.com/jeffersonrodriguezc/Forecasting_COVID_BGA/blob/main/images/logo_alcaldia.png?raw=true" width="350" height="250" />

Trabajo desarrollado para su estudio en la Secretaria de Salud de la Alcaldía de Bucaramanga.

## Evaluación de rendimiento de modelos de ML y DL para la predicción de casos de COVID-19 en Bucaramanga

#### Modelos explorados:
Los modelos siguieron dos estructuras o formas de uso: Recursivo y multi-modelo por días. Para los modelos de DL se exploró el uso de múltiples variables para los modelos ML solo univariables.

1. SVM para regresión
2. XGBoost (Multi-modelo por días mejor rendimiento)
3. RNN 
4. RNN Autoregresivo
5. Usando librerías como Greykite y Propnet

### Visualizaciones

### XGBoost - Multi-modelo por días - Sin suavizado
<img src="https://github.com/jeffersonrodriguezc/Forecasting_COVID_BGA/blob/main/images/Xgboost_directo_results.gif?raw=true](https://github.com/jeffersonrodriguezc/Forecasting_COVID_BGA/blob/main/images/Xgboost_directo_results.gif?raw=true" width="750" height="400" />

### XGBoost - Multi-modelo por días - Con suavizado
<img src="https://github.com/jeffersonrodriguezc/Forecasting_COVID_BGA/blob/main/images/xgboost_directo_smooth.gif?raw=true" width="750" height="400" />

### Forecastin usando Greykite y Propnet
<img src="https://github.com/jeffersonrodriguezc/Forecasting_COVID_BGA/blob/main/images/forecast_greykite.png?raw=true" width="750" height="400" />

## Efectos de factores socio-demográficos y comorbilidades en los fallecimiento por COVID-19

#### Interpretación & Análisis de Resultados

<img src="https://github.com/jeffersonrodriguezc/Forecasting_COVID_BGA/blob/main/images/impacto_model_1_shap.png?raw=true" width="450" height="200" />

<img src="https://github.com/jeffersonrodriguezc/Forecasting_COVID_BGA/blob/main/images/impacto_model_1_shap_mean.png?raw=true" width="450" height="200" />

<img src="https://github.com/jeffersonrodriguezc/Forecasting_COVID_BGA/blob/main/images/shap_values_interaction_tabla_comorbilidades.png?raw=true" width="450" height="200" />

<img src="https://github.com/jeffersonrodriguezc/Forecasting_COVID_BGA/blob/main/images/efecto_edad_fallecimiento.png?raw=true" width="450" height="200" />

<img src="https://github.com/jeffersonrodriguezc/Forecasting_COVID_BGA/blob/main/images/riesgo_fallecimiento__estrato.png?raw=true" width="750" height="400" />

<img src="https://github.com/jeffersonrodriguezc/Forecasting_COVID_BGA/blob/main/images/simulador_riesgo_fallecimiento.png?raw=true" width="750" height="400" />
