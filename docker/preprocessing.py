import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer


class WeatherFeatureBuilder(BaseEstimator, TransformerMixin):
    """
    Reproduce las transformaciones estructurales realizadas en el notebook.

    - Elimina columnas que no corresponden a inferencia.
    - Elimina Unnamed: 0 si aparece.
    - Convierte Date a datetime.
    - Crea Year, Month y Day.
    - Codifica RainToday como 0/1.
    - Crea Rainfall_log.
    - Conserva las columnas necesarias para el modelo.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Columna índice accidental
        if "Unnamed: 0" in X.columns:
            X = X.drop(columns=["Unnamed: 0"])

        # Columnas que no deben usarse para inferencia
        for col in ["RainTomorrow", "RainfallTomorrow"]:
            if col in X.columns:
                X = X.drop(columns=[col])

        # Date -> Year, Month, Day
        if "Date" in X.columns:
            X["Date"] = pd.to_datetime(X["Date"], errors="coerce")
            X["Year"] = X["Date"].dt.year
            X["Month"] = X["Date"].dt.month
            X["Day"] = X["Date"].dt.day
            X = X.drop(columns=["Date"])
        else:
            if "Month" not in X.columns:
                X["Month"] = np.nan
            if "Year" not in X.columns:
                X["Year"] = np.nan
            if "Day" not in X.columns:
                X["Day"] = np.nan

        # RainToday -> binaria
        if "RainToday" in X.columns:
            X["RainToday"] = X["RainToday"].map({
                "No": 0,
                "Yes": 1,
                "no": 0,
                "yes": 1,
                "NO": 0,
                "YES": 1,
                0: 0,
                1: 1,
                0.0: 0,
                1.0: 1
            })

        # Rainfall_log
        if "Rainfall" in X.columns:
            X["Rainfall_log"] = np.log1p(X["Rainfall"].clip(lower=0))
        else:
            X["Rainfall_log"] = np.nan

        columnas_requeridas = [
            "Location",
            "MinTemp",
            "MaxTemp",
            "Rainfall",
            "Evaporation",
            "Sunshine",
            "WindGustDir",
            "WindGustSpeed",
            "WindDir9am",
            "WindDir3pm",
            "WindSpeed9am",
            "WindSpeed3pm",
            "Humidity9am",
            "Humidity3pm",
            "Pressure9am",
            "Pressure3pm",
            "Cloud9am",
            "Cloud3pm",
            "Temp9am",
            "Temp3pm",
            "RainToday",
            "Year",
            "Month",
            "Day",
            "Rainfall_log"
        ]

        for col in columnas_requeridas:
            if col not in X.columns:
                X[col] = np.nan

        return X[columnas_requeridas]


class GeoSeasonalImputer(BaseEstimator, TransformerMixin):
    """
    Imputa valores faltantes replicando la lógica del notebook:

    1. Numéricas: mediana por Location + Month.
    2. Numéricas restantes: mediana de ciudades vecinas en el mismo Month.
    3. Categóricas: moda por Location + Month.
    4. Categóricas restantes: moda de ciudades vecinas en el mismo Month.
    5. Fallback final: mediana global o moda global aprendida en entrenamiento.

    Importante:
    vecinos_dict debe ser el diccionario construido en el notebook con NearestNeighbors.
    """

    def __init__(self, vecinos_dict=None):
        self.vecinos_dict = vecinos_dict or {}

        self.cols_numericas = [
            "MinTemp",
            "MaxTemp",
            "Rainfall",
            "Evaporation",
            "Sunshine",
            "WindGustSpeed",
            "WindSpeed9am",
            "WindSpeed3pm",
            "Humidity9am",
            "Humidity3pm",
            "Pressure9am",
            "Pressure3pm",
            "Cloud9am",
            "Cloud3pm",
            "Temp9am",
            "Temp3pm"
        ]

        self.cols_categoricas = [
            "WindGustDir",
            "WindDir9am",
            "WindDir3pm",
            "RainToday"
        ]

    @staticmethod
    def _moda(serie):
        serie = serie.dropna()
        if len(serie) == 0:
            return np.nan

        moda = serie.mode()
        if len(moda) == 0:
            return np.nan

        return moda.iloc[0]

    def fit(self, X, y=None):
        X = X.copy()

        # Estadísticos por Location + Month para numéricas
        self.num_loc_month_ = {}
        for col in self.cols_numericas:
            if col in X.columns:
                self.num_loc_month_[col] = (
                    X.groupby(["Location", "Month"])[col]
                    .median()
                )

        # Estadísticos globales para fallback
        self.num_global_ = {}
        for col in self.cols_numericas:
            if col in X.columns:
                self.num_global_[col] = X[col].median()

        # Moda por Location + Month para categóricas
        self.cat_loc_month_ = {}
        for col in self.cols_categoricas:
            if col in X.columns:
                self.cat_loc_month_[col] = (
                    X.groupby(["Location", "Month"])[col]
                    .agg(self._moda)
                )

        # Moda global para fallback
        self.cat_global_ = {}
        for col in self.cols_categoricas:
            if col in X.columns:
                self.cat_global_[col] = self._moda(X[col])

        # Guardamos copia de entrenamiento para poder consultar vecinos
        columnas_base = ["Location", "Month"] + self.cols_numericas + self.cols_categoricas
        columnas_base = [c for c in columnas_base if c in X.columns]
        self.train_reference_ = X[columnas_base].copy()

        return self

    def _imputar_numerica(self, X, col):
        if col not in X.columns:
            return X

        # 1. Mediana por misma Location + Month
        if col in self.num_loc_month_:
            keys = list(zip(X["Location"], X["Month"]))
            valores = pd.Series(
                [self.num_loc_month_[col].get(k, np.nan) for k in keys],
                index=X.index
            )
            X[col] = X[col].fillna(valores)

        # 2. Mediana por vecinos geográficos del mismo mes
        faltantes_idx = X.index[X[col].isna()].tolist()

        for idx in faltantes_idx:
            loc = X.at[idx, "Location"]
            month = X.at[idx, "Month"]

            vecinos = self.vecinos_dict.get(loc, [])

            valores_vecinos = self.train_reference_.loc[
                (self.train_reference_["Location"].isin(vecinos)) &
                (self.train_reference_["Month"] == month),
                col
            ].dropna()

            if len(valores_vecinos) > 0:
                X.at[idx, col] = valores_vecinos.median()

        # 3. Fallback global
        if col in self.num_global_:
            X[col] = X[col].fillna(self.num_global_[col])

        return X

    def _imputar_categorica(self, X, col):
        if col not in X.columns:
            return X

        # 1. Moda por misma Location + Month
        if col in self.cat_loc_month_:
            keys = list(zip(X["Location"], X["Month"]))
            valores = pd.Series(
                [self.cat_loc_month_[col].get(k, np.nan) for k in keys],
                index=X.index
            )
            X[col] = X[col].fillna(valores)

        # 2. Moda por vecinos geográficos del mismo mes
        faltantes_idx = X.index[X[col].isna()].tolist()

        for idx in faltantes_idx:
            loc = X.at[idx, "Location"]
            month = X.at[idx, "Month"]

            vecinos = self.vecinos_dict.get(loc, [])

            valores_vecinos = self.train_reference_.loc[
                (self.train_reference_["Location"].isin(vecinos)) &
                (self.train_reference_["Month"] == month),
                col
            ]

            moda_vecinos = self._moda(valores_vecinos)

            if not pd.isna(moda_vecinos):
                X.at[idx, col] = moda_vecinos

        # 3. Fallback global
        if col in self.cat_global_:
            X[col] = X[col].fillna(self.cat_global_[col])

        return X

    def transform(self, X):
        X = X.copy()

        for col in self.cols_numericas:
            X = self._imputar_numerica(X, col)

        for col in self.cols_categoricas:
            X = self._imputar_categorica(X, col)

        return X


class FinalColumnSelector(BaseEstimator, TransformerMixin):
    """
    Aplica la selección final de columnas usada en df_model.
    """

    def __init__(self):
        self.columnas_finales = [
            "Location",
            "MinTemp",
            "Sunshine",
            "WindGustDir",
            "WindGustSpeed",
            "WindDir9am",
            "WindDir3pm",
            "WindSpeed9am",
            "WindSpeed3pm",
            "Humidity9am",
            "Humidity3pm",
            "Pressure3pm",
            "Cloud9am",
            "Cloud3pm",
            "Temp3pm",
            "RainToday",
            "Month",
            "Rainfall_log"
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        for col in self.columnas_finales:
            if col not in X.columns:
                X[col] = np.nan

        return X[self.columnas_finales]


def crear_preprocesador(vecinos_dict=None):

    cat_cols = [
        "Location",
        "WindGustDir",
        "WindDir9am",
        "WindDir3pm",
        "Month"
    ]

    standard_cols = [
        "MinTemp",
        "Temp3pm",
        "Pressure3pm",
        "Humidity9am",
        "Humidity3pm",
        "Sunshine"
    ]

    robust_cols = [
        "Rainfall_log",
        "WindGustSpeed",
        "WindSpeed9am",
        "WindSpeed3pm"
    ]

    passthrough_cols = [
        "Cloud9am",
        "Cloud3pm",
        "RainToday"
    ]

    transformaciones_finales = ColumnTransformer(
        transformers=[
            (
                "standard",
                Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]),
                standard_cols
            ),
            (
                "robust",
                Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", RobustScaler())
                ]),
                robust_cols
            ),
            (
                "passthrough",
                Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="median"))
                ]),
                passthrough_cols
            ),
            (
                "cat",
                Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(
                        drop="first",
                        handle_unknown="ignore",
                        sparse_output=False
                    ))
                ]),
                cat_cols
            )
        ],
        verbose_feature_names_out=False
    )

    preprocesador = Pipeline(steps=[
        ("features", WeatherFeatureBuilder()),
        ("imputacion_geo_estacional", GeoSeasonalImputer(vecinos_dict=vecinos_dict)),
        ("seleccion_columnas", FinalColumnSelector()),
        ("transformaciones_finales", transformaciones_finales)
    ])

    return preprocesador