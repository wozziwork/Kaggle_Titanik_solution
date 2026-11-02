import pandas as pd

from config import config as cfg

from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold, cross_validate

from dataset import Dataset



def evaluate_model_cv(model : BaseEstimator, data : Dataset, scoring : dict, cv : StratifiedKFold) -> dict:
    """
    Валидация переданной модели по переданным аргументам.
    Исходная модель остается неизменной, на каждом фолде обучается ее копия
    Args:
         model: модель для валидации
         data: dataset с X_train, y_train
         scoring: словарь с метриками
         cv: схема разбиения данных для кросс-валидации
    Returns:
        cv_results: словарь с результатами кросс-валидации
    """
    cv_results = cross_validate(
        estimator=model,
        X=data.X_train,
        y=data.y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
    )
    return cv_results


def grid_search_summary(result : dict, best_index : int) -> pd.DataFrame:
    """
    Фильтрует таблицу с результатами GridSearchCV, оставляет только средние валидационные метрики из конфига по индексу лучшего эксперимента
    Args:
        result: словарь с результатами
        best_index: индекс лучшего сочетания гиперпараметров
    Returns:
        summary: Датафрейм, где строки - средние валидационные метрики и среднее время обучения для лучшего сочетания гиперпараметров,
                                         столбец - название эксперимента
    """
    cols =['mean_fit_time'] + [f'mean_test_{metric}' for metric in cfg.validation.scorers]
    summary = pd.DataFrame(result).loc[[best_index], cols]
    summary.index = [cfg.general.experiment_name]
    return summary.T