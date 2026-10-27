
import pandas as pd

from omegaconf import OmegaConf
from config import config as cfg

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from validation import (
    build_scoring, build_cv, build_grid_search, evaluate_model_cv, grid_search_summary, run_boosting_experiment
)
from preprocessing import build_preprocessor, build_catboost_preprocessor, build_category_caster_preprocessor
from artifacts import save_model, save_experiment_log
from dataset import Dataset



def fit_and_evaluate(model : Pipeline, data : Dataset) -> tuple[Pipeline, pd.DataFrame]:
    """
    Оценка модели и ее обучение
    Args:
        model: исходная модель
        data: датасет с X_train, y_train
    Returns:
        model: обученная модель
        cv_results: DataFrame с результатами по фолдам и строкой средних значений
    """
    scoring = build_scoring()
    cv = build_cv()

    cv_results = pd.DataFrame(evaluate_model_cv(model, data, scoring, cv))
    cv_results.loc['mean'] = cv_results.mean()

    model.fit(data.X_train, data.y_train)
    return model, cv_results


def search_and_fit(model : Pipeline, data : Dataset, model_name : str) -> tuple[Pipeline, dict, int]:
    """
    Перебор гиперпараметров модели, заданных в конфиге.
    Возвращает модель с лучшим результатом, а также результаты поиска
    Args:
        model: исходная модель
        data: датасет с X_train, y_train
        model_name: имя модели
    Returns:
        best_model: обученная на всех исходных данных модель с лучшим, среди проверенных по выбранной метрике сочетанием, гиперпараметров
        search_results: словарь с результатами GridSearch
        best_index: индекс лучшего сочетания гиперпараметров
    """
    scoring = build_scoring()
    cv = build_cv()

    # достаю параметры для GridSearchCV из конфига вида ИмяШагаПайплайна__параметр : [значения]
    param_grid = OmegaConf.to_container(cfg.param_grids.get(model_name,{}))
    grid_search = build_grid_search(model, param_grid, scoring, cv)
    grid_search.fit(data.X_train, data.y_train)

    best_model = grid_search.best_estimator_
    search_results = grid_search.cv_results_
    best_index = grid_search.best_index_

    return best_model, search_results, best_index



def train(data : Dataset) -> dict[str, Pipeline]:
    """
    Создает, оценивает и обучает модели, выбранные в конфиге
    Args:
        data: датасет с X_train и y_train
    Returns:
        trained_models: словарь {имя_модели : обученный пайплайн}
    """
    trained_models = {}
    results = {}

    if cfg.to_train.logistic_regression:
        model_name  = 'logistic_regression'
        base_preprocessor = build_preprocessor()
        logreg = Pipeline([
            ('preprocessor', base_preprocessor),
            ('model', LogisticRegression(**cfg.models.logistic_regression)),
        ])

        if cfg.to_grid_search.logistic_regression:
            logreg, experiment_results, best_index = search_and_fit(logreg, data, model_name)

            # саммари отображает mean по каждой метрике из конфига (для лучшего эксперимента)
            if cfg.search.summarize_results:
                experiment_results = grid_search_summary(experiment_results, best_index)
        else:
            logreg, experiment_results = fit_and_evaluate(logreg, data)

        trained_models[model_name] = logreg
        results[model_name] = experiment_results



    if cfg.to_train.knn:
        base_preprocessor = build_preprocessor()
        model_name = 'knn'
        knn = Pipeline([
            ('preprocessor', base_preprocessor),
            ('model', KNeighborsClassifier(**cfg.models.knn)),
        ])

        if cfg.to_grid_search.knn:
            knn, experiment_results, best_index = search_and_fit(knn, data, model_name)

            # саммари отображает mean по каждой метрике из конфига (для лучшего эксперимента)
            if cfg.search.summarize_results:
                experiment_results = grid_search_summary(experiment_results, best_index)
        else:
            knn, experiment_results = fit_and_evaluate(knn, data)

        trained_models[model_name] = knn
        results[model_name] = experiment_results

    if cfg.to_train.decision_tree:
        base_preprocessor = build_preprocessor(scale_numeric=False)
        model_name = 'decision_tree'
        decision_tree = Pipeline([
            ('preprocessor', base_preprocessor),
            ('model', DecisionTreeClassifier(**cfg.models.decision_tree)),
        ])

        if cfg.to_grid_search.decision_tree:
            decision_tree, experiment_results, best_index = search_and_fit(decision_tree, data, model_name)

            # саммари отображает mean по каждой метрике из конфига (для лучшего эксперимента)
            if cfg.search.summarize_results:
                experiment_results = grid_search_summary(experiment_results, best_index)
        else:
            decision_tree, experiment_results = fit_and_evaluate(decision_tree, data)
        trained_models[model_name] = decision_tree
        results[model_name] = experiment_results

    if cfg.to_train.random_forest:
        base_preprocessor = build_preprocessor(scale_numeric=False)
        model_name = 'random_forest'
        random_forest = Pipeline([
            ('preprocessor', base_preprocessor),
            ('model', RandomForestClassifier(**cfg.models.random_forest)),
        ])

        if cfg.to_grid_search.random_forest:
            random_forest, experiment_results, best_index = search_and_fit(random_forest, data, model_name)
            if cfg.search.summarize_results:
                experiment_results = grid_search_summary(experiment_results, best_index)
        else:
            random_forest, experiment_results = fit_and_evaluate(random_forest, data)
        trained_models[model_name] = random_forest
        results[model_name] = experiment_results

    if cfg.to_train.catboost:
        model_name = 'catboost'
        # преобразуем cat_features в кортеж для совместимости клонирования модели
        catboost_params = OmegaConf.to_container(cfg.models.catboost, resolve=True)
        catboost_params['cat_features'] = tuple(catboost_params['cat_features'])

        catboost = CatBoostClassifier(**catboost_params)
        catboost_preprocessor = build_catboost_preprocessor()

        catboost, experiment_results, chosen_iterations = run_boosting_experiment(model_name, catboost, data, build_catboost_preprocessor)

        # Обучаем на всех тренировочных данных без eval_set
        # число итераций выбрано по CV, поэтому early stopping отключаем.
        catboost.set_params(early_stopping_rounds=None, use_best_model=False, iterations=chosen_iterations)
        X_train = catboost_preprocessor.fit_transform(data.X_train)
        y_train = data.y_train

        catboost.fit(X_train, y_train)
        catboost = Pipeline([
            ('preprocessor', catboost_preprocessor),
            ('model', catboost),
        ])

        trained_models[model_name] = catboost
        results[model_name] = experiment_results

    if cfg.to_train.xgboost:
        model_name = 'xgboost'
        xgboost_params = OmegaConf.to_container(cfg.models.xgboost, resolve=True)
        xgboost = XGBClassifier(**xgboost_params)
        xgboost_preprocessor = build_category_caster_preprocessor()

        xgboost, experiment_results, chosen_iterations = run_boosting_experiment(model_name, xgboost, data, build_category_caster_preprocessor)

        # Обучаем на всех тренировочных данных без eval_set
        # число итераций выбрано по CV, поэтому early stopping отключаем.
        xgboost.set_params(early_stopping_rounds=None, n_estimators=chosen_iterations)
        X_train = xgboost_preprocessor.fit_transform(data.X_train)
        y_train = data.y_train

        xgboost.fit(X_train, y_train)
        xgboost = Pipeline([
            ('preprocessor', xgboost_preprocessor),
            ('model', xgboost),
        ])

        trained_models[model_name] = xgboost
        results[model_name] = experiment_results

    if cfg.to_train.lightgbm:
        model_name = 'lightgbm'
        lgbm_params = OmegaConf.to_container(cfg.models.lightgbm, resolve=True)
        lgbm = LGBMClassifier(**lgbm_params)
        lgbm_preprocessor = build_category_caster_preprocessor()

        lgbm, experiment_results, chosen_iterations = run_boosting_experiment(model_name, lgbm, data, build_category_caster_preprocessor)

        # Обучаем на всех тренировочных данных без eval_set
        # число итераций выбрано по CV, поэтому early stopping отключаем.
        lgbm.set_params(early_stopping_round=0, n_estimators=chosen_iterations)
        X_train = lgbm_preprocessor.fit_transform(data.X_train)
        y_train = data.y_train

        lgbm.fit(X_train, y_train)
        lgbm = Pipeline([
            ('preprocessor', lgbm_preprocessor),
            ('model', lgbm),
        ])

        trained_models[model_name] = lgbm
        results[model_name] = experiment_results

    if cfg.general.save_models:
        for model_name, model in trained_models.items():
            save_model(model, model_name)

    if cfg.general.save_logs:
        for model_name, model in trained_models.items():
            save_experiment_log(model, results[model_name], model_name)

    for model_name, experiment_results in results.items():
        print(f'Result for {model_name}')
        print(pd.DataFrame(experiment_results).to_string() + '\n')


    return trained_models
