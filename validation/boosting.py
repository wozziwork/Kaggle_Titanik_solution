import pandas as pd
import lightgbm
import copy
import numpy as np

from config import config as cfg
from omegaconf import OmegaConf

from validation.general import build_cv

from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid
from sklearn.compose import ColumnTransformer
from sklearn.base import clone

from collections.abc import Callable
from statistics import mean
from statistics import median

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from dataset import Dataset
from preprocessing import CategoryCaster, build_catboost_preprocessor, build_category_caster_preprocessor

BoostingModel = CatBoostClassifier | XGBClassifier | LGBMClassifier

def evaluate_boosting_cv(
        boosting_model : BoostingModel,
        data : Dataset,
        build_preprocessor_func : Callable[[], ColumnTransformer | CategoryCaster],
    ) -> list[dict]:
    """
    Кросс-валидация для бустингов с early stopping.
    Args:
        boosting_model: исходная модель, клонируемая для каждого фолда
        data: данные для обучения и валидации
        build_preprocessor_func: функция, создающая и возвращающая препроцессор для бустинга.
    Returns:
        results: список словарей, где содержится val_accuracy, train_accuracy, trees_cnt (число итераций выбранной модели)
    """
    cv = build_cv()
    results = []
    fit_params = {}


    if isinstance(boosting_model, XGBClassifier):
        fit_params['verbose'] = cfg.general.boosting_verbose
    elif isinstance(boosting_model, LGBMClassifier):
        fit_params['callbacks'] = [lightgbm.log_evaluation(period = cfg.general.boosting_verbose)]


    for train_idx, val_idx in cv.split(data.X_train, data.y_train):
        X_train, X_val = data.X_train.iloc[train_idx], data.X_train.iloc[val_idx]
        y_train, y_val = data.y_train.iloc[train_idx], data.y_train.iloc[val_idx]

        preprocessor =  build_preprocessor_func()
        X_train = preprocessor.fit_transform(X_train)
        X_val = preprocessor.transform(X_val)

        if isinstance(boosting_model, LGBMClassifier):
            eval_params = {
                'eval_X' : X_val,
                'eval_y' : y_val
            }
        else:
            eval_params = {'eval_set' : [(X_val, y_val)]}

        fold_model = clone(boosting_model)
        fold_model.fit(
            X_train, y_train,
            **eval_params,
            **fit_params
        )
        predictions = fold_model.predict(X_val)
        train_predictions = fold_model.predict(X_train)

        if isinstance(fold_model, CatBoostClassifier):
            trees_cnt = fold_model.tree_count_
        elif isinstance(fold_model, XGBClassifier):
            if fold_model.get_params().get('early_stopping_rounds'):
                trees_cnt = fold_model.best_iteration + 1
            else:
                trees_cnt = fold_model.get_booster().num_boosted_rounds()
        elif isinstance(fold_model, LGBMClassifier):
            if fold_model.get_params().get('early_stopping_round', 0):
                trees_cnt = fold_model.best_iteration_
            else:
                trees_cnt = fold_model.n_estimators_
        else:
            raise TypeError('Ожидался CatboostClassifier, XGBClassifier или LGBMClassifier')

        results.append(
            {
                'val_accuracy' : accuracy_score(y_val, predictions),
                'train_accuracy' : accuracy_score(y_train, train_predictions),
                'trees_cnt' : trees_cnt
            }
        )

    return results

def search_boosting_cv(
        model : BoostingModel,
        data : Dataset,
        param_grid : dict[str, list],
        build_preprocessor_func : Callable[[], ColumnTransformer | CategoryCaster]
    ) -> tuple[dict[str, int | float | str | bool], list[dict]]:
    """
    Поиск по сетке для бустингов с кросс-валидацией
    Args:
        model: исходная модель, клонируемая для каждого сочетания параметров
        data: данные для обучения и валидации (разбиваются на фолды внутри CV)
        param_grid: сетка параметров для перебора
        build_preprocessor_func: функция, создающая и возвращающая препроцессор для бустинга

    Returns:
        best_params: сочетание параметров с наибольшей средней accuracy по фолдам
        grid_search_results: список словарей с параметрами, средними accuracy для train, validation и результатами по фолдам для каждого сочетания
    """
    grid_search_results = []
    best_mean_val_accuracy = -1
    best_params = None
    param_grid = {name.removeprefix('model__') : param_values for name, param_values in param_grid.items()}
    for params in ParameterGrid(param_grid):
        candidate = clone(model)
        candidate.set_params(**params)
        #получаем результаты для конкретных параметров
        fold_results = evaluate_boosting_cv(candidate, data, build_preprocessor_func)

        mean_val_accuracy = mean(res['val_accuracy'] for res in fold_results)
        mean_train_accuracy = mean(res['train_accuracy'] for res in fold_results)
        grid_search_results.append({
            'params': params,
            'mean_val_accuracy' : mean_val_accuracy,
            'mean_train_accuracy' : mean_train_accuracy,
            'fold_results' : fold_results
        })

        if mean_val_accuracy > best_mean_val_accuracy:
            best_mean_val_accuracy = mean_val_accuracy
            best_params = params

    return best_params, grid_search_results


def summarize_boosting_cv(cv_results : list[dict]) -> tuple[pd.DataFrame, int]:
    """
    Вычисляет медианное число итераций бустинга на CV.
    В результаты добавляет строку со средними val и train accuracy.
    Args:
        cv_results: список словарей с результатами кросс-валидации бустинга
    Returns:
        summary_results: DataFrame с результатами по фолдам и строкой 'mean' со средними val_accuracy и train_accuracy.
        chosen_iterations: медианное число итераций бустинга на CV

    """
    chosen_iterations = int(median(res['trees_cnt'] for res in cv_results))
    summary_results = pd.DataFrame(cv_results)
    summary_results.loc['mean', 'val_accuracy'] = summary_results['val_accuracy'].mean()
    summary_results.loc['mean', 'train_accuracy'] = summary_results['train_accuracy'].mean()
    return summary_results, chosen_iterations


def summarize_boosting_search_cv(search_cv_results : list[dict]) -> tuple[pd.DataFrame, int]:
    """
    Вычисляет медианное число итераций для победителя по результатам поиска по сетке.
    Победитель выбирается по средней валидационной accuracy.
    В результаты добавляет две строки со средними train/val accuracy на CV для победителя и с параметрами победителя.
    Args:
        search_cv_results: список словарей с результатами кросс-валидации для каждого сочетания параметров бустинга
    Returns:
          summary_results: DataFrame с результатами поиска по сетке и добавленными строками 'mean_for_best', 'best_params'
          chosen_iterations: медианное число итераций для лучшего бустинга по результатам поиска по сетке
    """
    # Выбираем сочетание параметров с наибольшей средней accuracy по фолдам
    best_result = max(search_cv_results, key=lambda result: result['mean_val_accuracy'])
    best_params = best_result['params']

    # Для итогового обучения берём медиану числа деревьев по фолдам best_result
    fold_results = best_result['fold_results']
    chosen_iterations = int(median(fold_res['trees_cnt'] for fold_res in fold_results))

    summary_results = pd.DataFrame(search_cv_results)
    summary_results.loc['mean_for_best', ['mean_val_accuracy', 'mean_train_accuracy']] = [
        best_result['mean_val_accuracy'],
        best_result['mean_train_accuracy'],
    ]
    summary_results.loc['best_params'] = None
    summary_results.at['best_params', 'params'] = best_params
    return summary_results, chosen_iterations

def run_boosting_experiment(
        model_name : str,
        model : BoostingModel,
        data : Dataset,
        build_preprocessor_func : Callable[[], ColumnTransformer | CategoryCaster]
    ) -> tuple[BoostingModel, pd.DataFrame, int]:
    """
    Согласно конфигу выполняет CV или поиск по сетке с CV для переданного бустинга.
    Исходная модель не меняется
    Args:
        model_name: ключ модели в конфиге
        model: исходная модель, клонируемая внутри функции
        data: тренировочные данные
        build_preprocessor_func: функция создания препроцессора
    Returns:
        model: необученная копия исходной модели, при поиске получает лучшие параметры
        experiment_results: DataFrame с результатами эксперимента
        chosen_iterations: медиана числа итераций по фолдам для используемого сочетания параметров
    """
    model = clone(model)
    if cfg.to_grid_search[model_name]:
        param_grid = OmegaConf.to_container(cfg.param_grids[model_name], resolve=True)
        best_params, experiment_results = search_boosting_cv(model, data, param_grid, build_preprocessor_func)
        experiment_results, chosen_iterations = summarize_boosting_search_cv(experiment_results)
        model.set_params(**best_params)
    else:
        experiment_results = evaluate_boosting_cv(model, data, build_preprocessor_func)
        experiment_results, chosen_iterations = summarize_boosting_cv(experiment_results)
    return model, experiment_results, chosen_iterations

def get_boosting_oof(
        boosting : CatBoostClassifier | LGBMClassifier,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> np.ndarray:
    """
    Получение oof проб для настроенных catboost и lgbm моделей.
    """
    oof_probs = np.full(X.shape[0], np.nan)
    cv = build_cv()

    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val =X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val =y.iloc[train_idx], y.iloc[val_idx]

        if isinstance(boosting, LGBMClassifier):
            preprocessor = build_category_caster_preprocessor()
        elif isinstance(boosting, CatBoostClassifier):
            preprocessor = build_catboost_preprocessor()
        else:
            raise ValueError("Нужен catboost или lgbm")

        X_train = preprocessor.fit_transform(X_train)
        X_val = preprocessor.transform(X_val)
        fold_model = copy.deepcopy(boosting)
        fold_model.fit(
            X_train, y_train
        )
        oof_probs[val_idx] = fold_model.predict_proba(X_val)[:,1]
    return oof_probs