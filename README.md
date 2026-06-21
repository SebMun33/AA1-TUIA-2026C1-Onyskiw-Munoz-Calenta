# Inferencia del modelo RainTomorrow con Docker

Este contenedor ejecuta la inferencia de un modelo de clasificación binaria para predecir la variable `RainTomorrow`.

El modelo recibe un archivo `.csv` con datos meteorológicos nuevos, aplica el mismo preprocesamiento utilizado durante el entrenamiento y genera un archivo de salida con:

- `probabilidad_lluvia`
- `prediccion_RainTomorrow`

## Archivos incluidos

- `Dockerfile`: instrucciones para construir la imagen.
- `requirements.txt`: librerías necesarias para ejecutar la inferencia.
- `inferencia.py`: script principal de inferencia.
- `preprocessing.py`: funciones y pipeline de preprocesamiento.
- `modelo_red_neuronal.keras`: modelo entrenado.
- `preprocesador.joblib`: pipeline de preprocesamiento ajustado con train.
- `columnas_modelo.joblib`: columnas esperadas por el modelo.
- `threshold.json`: umbral elegido para transformar probabilidades en clases.

## Construcción de la imagen

Desde la carpeta donde está el Dockerfile:

```bash
docker build -t lluvia-inferencia .