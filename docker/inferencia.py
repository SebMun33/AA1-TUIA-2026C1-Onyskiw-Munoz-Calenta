import sys
import json
from pathlib import Path

import joblib
import pandas as pd
import tensorflow as tf


BASE_DIR = Path(__file__).resolve().parent


def resolver_path(path_str):
    path = Path(path_str)
    if path.is_absolute():
        return path
    return BASE_DIR / path


def main():
    if len(sys.argv) != 3:
        print("Uso: python inferencia.py input.csv output.csv")
        sys.exit(1)

    input_path = resolver_path(sys.argv[1])
    output_path = resolver_path(sys.argv[2])

    modelo_path = BASE_DIR / "modelo_red_neuronal.keras"
    preprocesador_path = BASE_DIR / "preprocesador.joblib"
    columnas_path = BASE_DIR / "columnas_modelo.joblib"
    threshold_path = BASE_DIR / "threshold.json"

    modelo = tf.keras.models.load_model(modelo_path)
    preprocesador = joblib.load(preprocesador_path)
    columnas_modelo = joblib.load(columnas_path)

    with open(threshold_path, "r", encoding="utf-8") as f:
        threshold = json.load(f)["threshold"]

    df = pd.read_csv(input_path)

    nombres_columnas_preprocesador = (
        preprocesador
        .named_steps["transformaciones"]
        .get_feature_names_out()
    )

    X_procesado = pd.DataFrame(
        X_procesado,
        columns=nombres_columnas_preprocesador
    )

    X_proc = X_proc.reindex(columns=columnas_modelo, fill_value=0)
    X_proc = X_proc.astype("float32")

    proba_lluvia = modelo.predict(X_proc).ravel()
    prediccion = (proba_lluvia >= threshold).astype(int)

    salida = pd.DataFrame({
        "probabilidad_lluvia": proba_lluvia,
        "prediccion_RainTomorrow": prediccion
    })

    salida.to_csv(output_path, index=False)

    print(f"Inferencia finalizada. Resultados guardados en: {output_path}")


if __name__ == "__main__":
    main()