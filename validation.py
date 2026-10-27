import optuna
from omegaconf import OmegaConf

from config import config as cfg

import pandas as pd
import lightgbm

from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
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

from preprocessing import CategoryCaster
from dataset import Dataset
from train.train_mlp import train_mlp_fold
from train.train_mlp_emb import train_mlp_emb_fold

BoostingModel = CatBoostClassifier | XGBClassifier | LGBMClassifier

def build_scoring() -> dict:
    """
    Создает и возвращает словарь метрик, выбранных в конфиге
    """
    available_scorers = {
        'accuracy': 'accuracy',
        'precision': make_scorer(precision_score, zero_division=0),
        'recall': make_scorer(recall_score, zero_division=0),
        'f1': make_scorer(f1_score, zero_division=0),
        'roc_auc': 'roc_auc',
    }
    return {scorer : available_scorers[scorer] for scorer in cfg.validation.scorers}


def build_cv() -> StratifiedKFold:
    """
    Создает и возвращает объект StratifiedKFold с настройками из конфига
    """
    cv = StratifiedKFold(
        n_splits=cfg.validation.n_splits,
        shuffle=cfg.validation.shuffle,
        random_state=cfg.general.seed if cfg.validation.shuffle else None
    )
    return cv


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


def build_grid_search(
        model : BaseEstimator,
        param_grid : dict[str,list[int | float | str | bool]],
        scoring,
        cv
    ) -> GridSearchCV:
    """
    Создает объект GridSearchCV, настроенный по переданным аргументам и конфигу. Поиск не происходит
    Args:
         model: модель или пайплайн для валидации
         param_grid: словарь с названиями гиперпараметров и списками значений для перебора
         scoring: словарь с метриками
         cv: схема разбиения данных для кросс-валидации
    Returns:
        Настроенный объект GridSearchCV
    """
    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        refit=cfg.search.refit,
        n_jobs=cfg.search.n_jobs,
        verbose=cfg.search.verbose,
        return_train_score=True
    )
    return search


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

def evaluate_boosting_cv(
        boosting_model : BoostingModel,
        data : Dataset,
        build_preprocessor_func : Callable[[], ColumnTransformer | CategoryCaster]
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
            trees_cnt = fold_model.best_iteration + 1
        elif isinstance(fold_model, LGBMClassifier):
            trees_cnt = fold_model.best_iteration_
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
    Исходная модель не меняется.
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


def mlp_cv(
        data,
        model_name,
        **params,
        ):
    X, y = data.X_train, data.y_train
    cv = build_cv()
    experiment_results = {}
    fold_n = 0
    full_history = []
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if model_name == 'mlp_emb':
            train_func = train_mlp_emb_fold
        elif model_name == 'mlp':
            train_func = train_mlp_fold
        else:
            raise ValueError(f'Неизвестное имя модели: {model_name}')

        model, preprocessor, history, _, best_epoch = train_func(
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val, **params)
        full_history.append(history)

        fold_n += 1
        experiment_results[fold_n] = {
            'best_epoch': best_epoch + 1,
            'best_val_loss': history['val_loss'][best_epoch],
            'val_acc_at_best_loss': history['val_accuracy'][best_epoch],
            'last_5_val_accuracy' : history['val_accuracy'][-5:],
            'last_5_train_loss': history['train_loss'][-5:],
            'last_5_val_loss': history['val_loss'][-5:],
        }
    mean_accuracy = mean(fold_res['val_acc_at_best_loss'] for fold_res in experiment_results.values())
    return experiment_results, full_history, mean_accuracy

def search_mlp_cv(data, model_name, param_grid, base_params):
    best_accuracy = -1
    best_params = None
    search_results = {}
    iteration_cnt = 0
    for params in ParameterGrid(param_grid):
        candidate_params = {**base_params, **params}
        experiment_results, _, mean_accuracy = mlp_cv(data, model_name, **candidate_params)

        search_results[iteration_cnt] = {
            'params' : candidate_params,
            'mean_accuracy' : mean_accuracy,
        }
        iteration_cnt+=1

        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            best_params = candidate_params

    return best_accuracy, best_params, search_results

def search_mlp_emb_optuna(data, model_name, base_params, n_trails=20):
    def objective(trial):
        n_layers = trial.suggest_int('n_layers', 1, 5)
        hidden_dims = [trial.suggest_categorical(f'width_{i}',[16, 32, 64, 128, 256]) for i in range(n_layers)]
        embedding_dims = [trial.suggest_categorical(f'emb_width_{i}', [4, 8, 12, 16, 20, 24]) for i in range(4)]

        params = {
            **base_params,
            'lr' : trial.suggest_float('lr', 0.0005, 0.003, log=True),
            'dropout': trial.suggest_float('dropout', 0.0, 0.4),
            'batch_size': trial.suggest_categorical('batch_size', [16, 24, 32, 64, 128]),
            'hidden_dims' : hidden_dims,
            'embedding_dims' : embedding_dims,
            'optimizer_name' : trial.suggest_categorical('optimizer_name', ['adam', 'adamw']),
            'weight_decay': trial.suggest_float('weight_decay', 0.0, 0.4),
            'activation_func': trial.suggest_categorical('activation_func', ['relu', 'leacky_relu']),

        }
        _, _, mean_accuracy = mlp_cv(data, model_name, **params)
        return mean_accuracy

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=cfg.general.seed)
    )
    study.optimize(objective, n_trials=n_trails)
    return study