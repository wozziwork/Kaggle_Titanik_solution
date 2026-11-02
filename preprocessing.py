import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import torch

from config import config as cfg


def build_preprocessor(scale_numeric : bool = True, sparse_output : bool = True) -> ColumnTransformer:
    """
    Создает препроцессор для числовых и категориальных признаков
    Args:
        scale_numeric : флаг, отвечающий за наличие стандартизации
        sparse_output : флаг, отвечающий за формат выхода OneHotEncoder.
            True - разреженная матрица, False - обычный numpy массив
    Returns: необученный настроенный ColumnTransformer
    """

    scaler = StandardScaler() if scale_numeric else 'passthrough'

    base_numeric_processing = Pipeline([
        ('imputer', SimpleImputer(strategy=cfg.preprocessing.numeric.strategy)),
        ('scaler', scaler),
    ])

    base_categorical_processing = Pipeline([
        ('imputer', SimpleImputer(strategy=cfg.preprocessing.categorical.strategy,
                                  fill_value=cfg.preprocessing.categorical.fill_value)),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=sparse_output)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', base_numeric_processing, list(cfg.dataset.numeric_columns)),
            ('categorical', base_categorical_processing, list(cfg.dataset.categorical_columns)),
        ]
    )
    return preprocessor


def build_catboost_preprocessor() -> ColumnTransformer:
    """
    Создает препроцессор для catboost, заполняющий пропуски в категориальных признаках.
    Остальные столбцы передаются без преобразований.
    Returns: необученный настроенный ColumnTransformer, возвращающий DataFrame после трансформации
    """

    categorical_processing = Pipeline([
        ('imputer', SimpleImputer(strategy=cfg.preprocessing.categorical.strategy,
                                  fill_value=cfg.preprocessing.categorical.fill_value))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('categorical', categorical_processing, list(cfg.models.catboost.cat_features)),
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    ).set_output(transform='pandas')

    return preprocessor


class CategoryCaster(TransformerMixin, BaseEstimator):
    """
    Преобразует указанные категориальные столбцы DataFrame в тип 'category'.
    При fit запоминает категории из обучающих данных.
    При transform использует сохраненные категории. Неизвестные значения преобразуются в NaN.
    Остальные столбцы и исходный DataFrame не меняются
    Args:
        cat_cols: список названий категориальных столбцов
    """

    def __init__(self, cat_cols : list):
        self.category_dtypes_ = None
        self.cat_cols = cat_cols

    def fit(self, X : pd.DataFrame, y : pd.Series | None = None):
        self.category_dtypes_ = {
            col : pd.CategoricalDtype(categories=X[col].dropna().unique()) for col in self.cat_cols
        }
        return self

    def transform(self, X : pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, dtype in self.category_dtypes_.items():
            X[col] = X[col].astype(dtype)
        return X

def build_category_caster_preprocessor() -> CategoryCaster:
    return CategoryCaster(list(cfg.dataset.categorical_columns))

def build_embedding_preprocessor() -> ColumnTransformer:
    """
    Создает препроцессор для MLPEmb.
    Заполняет пропуски, стандартизует числовые признаки и кодирует категории целочисленными индексами.
    Returns: необученный настроенный ColumnTransformer, возвращающий NumPy-массив после трансформации
    """
    numerical_processing = Pipeline([
        ('imputer', SimpleImputer(strategy=cfg.preprocessing.numeric.strategy)),
        ('scaler', StandardScaler()),
    ])

    categorical_processing = Pipeline([
        ('imputer', SimpleImputer(strategy=cfg.preprocessing.categorical.strategy,
                                  fill_value=cfg.preprocessing.categorical.fill_value
        )),
        ('encoder', OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=-1,
            dtype=np.int64
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', numerical_processing, list(cfg.dataset.numeric_columns)),
            ('categorical', categorical_processing, list(cfg.dataset.categorical_columns)),
        ]
    )
    return preprocessor

def prepare_embedding_inputs(X_processed, device):
    """
    Разбивает X_processed на два тензора, содержащих по порядку сначала численные(float32),
        а затем закодированные категориальные признаки(int32)
    Ко всем значениям категориальных признаков добавляется 1 (под неизвестную категорию)
    """
    n_num = len(cfg.dataset.numeric_columns)
    X_num = torch.from_numpy(X_processed[:, :n_num].astype('float32')).to(device)
    X_cat = torch.from_numpy(X_processed[:, n_num:].astype('int32') + 1).to(device)
    return X_num, X_cat