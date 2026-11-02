import pandas as pd

from omegaconf import OmegaConf
from artifacts import save_dl_experiment_results, save_dl_checkpoint
from dataset import Dataset
from train.train_mlp import train_mlp_fold
from train.train_mlp_emb import train_mlp_emb_fold
from visualization import show_results_plot
from config import config as cfg
from validation.neural import mlp_cv, search_mlp_cv





def get_params(name : str) -> dict:
    """
    Достает из конфига настройки архитектуры модели и ее обучения и объединяет их в словарь.
    """
    return {**OmegaConf.to_container(cfg.models[name], resolve=True), **OmegaConf.to_container(cfg.dl_train[name], resolve=True)}

def train_dl(
        dataset : Dataset,
        oof_probs : dict
    ) -> tuple[dict, dict, dict]:
    """
    Проведение кросс-валидации или поиска по сетке для MLP, MLPEmb моделей (настраивается в конфиге).
    Вывод и сохранение логов экспериментов, вывод графиков обучения (настраивается в конфиге).
    Сохраняет обученные на всем трейне модели и их препроцессоры в saved_models (настраивается в конфиге).
    При обычной CV для MLPEmb также сохраняет модели и препроцессоры всех фолдов, если включено сохранение моделей.
    Args:
        dataset : класс-датасет с данными для обучения
        oof_probs : словарь с oof вероятностями (дополняется вероятностями MLP, MLPEmb при обычной CV)
    Returns:
        oof_probs : заполненный словарь с oof вероятностями
        results : словарь {имя_модели: {'model':обученная модель,'preprocessor':обученный препроцессор}}
        cv_scores : словарь {имя_модели: средняя accuracy на кросс-валидации}.
            Заполняется только для моделей, обучаемых без grid search
    """
    results = {}
    cv_scores = {}
    if cfg.to_train.mlp:
        model_name = 'mlp'
        params = get_params(model_name)

        if cfg.to_grid_search.mlp:
            param_grid = OmegaConf.to_container(cfg.param_grids.mlp, resolve=True)
            # params - лучшее сочетание параметров по поиску по сетке
            best_acc, params, search_results = search_mlp_cv(dataset, model_name, param_grid, params)
            experiment_results = pd.DataFrame(search_results).T
            experiment_results.loc['best_params'] = None
            experiment_results.at['best_params', 'params'] = params
            experiment_results.loc['best_accuracy', 'mean_accuracy'] = best_acc
        else:
            cv_results, full_history, mean_accuracy, ensemble, mlp_oof_probs = mlp_cv(
                dataset.X_train,
                dataset.y_train,
                model_name,
                **params
            )
            oof_probs[model_name] = mlp_oof_probs
            experiment_results = pd.DataFrame(cv_results).T
            experiment_results.index.name = f'fold_{model_name}'
            experiment_results.loc['mean_accuracy', 'val_acc_at_best_epoch'] = mean_accuracy
            cv_scores[model_name] = mean_accuracy
            if cfg.general.show_graphs:
                for n, history in enumerate(full_history, start=1):
                    show_results_plot(history, n)

        if cfg.general.train_dl_on_all_data:
            model, preprocessor, _, checkpoint, _ = train_mlp_fold(
                X_train=dataset.X_train,
                y_train=dataset.y_train,
                **params
            )
            results[model_name] = {'model': model, 'preprocessor': preprocessor}

            if cfg.general.save_models:
                save_dl_checkpoint(checkpoint, preprocessor, model_name)

        if cfg.general.show_logs:
            print(experiment_results.to_string())

        if cfg.general.save_logs:
            save_dl_experiment_results(model_name, experiment_results)

    if cfg.to_train.mlp_emb:
        model_name = 'mlp_emb'
        params = get_params(model_name)

        if cfg.to_grid_search[model_name]:
            param_grid = OmegaConf.to_container(cfg.param_grids[model_name], resolve=True)
            # params - лучшее сочетание параметров по поиску по сетке
            best_acc, params, search_results = search_mlp_cv(dataset, model_name, param_grid, params)
            experiment_results = pd.DataFrame(search_results).T
            experiment_results.loc['best_params'] = None
            experiment_results.at['best_params', 'params'] = params
            experiment_results.loc['best_accuracy', 'mean_accuracy'] = best_acc
        else:
            # ensemble - список кортежей (модель, препроцессор, чекпоинт)
            cv_results, full_history, mean_accuracy, ensemble, mlpemb_oof_probs = mlp_cv(
                dataset.X_train,
                dataset.y_train,
                model_name,
                **params
            )
            if cfg.to_train.mlp_emb_fold_ensemble and cfg.general.save_models:
                for fold_n, (_, fold_preprocessor, fold_checkpoint) in enumerate(ensemble, start=1):
                    save_dl_checkpoint(fold_checkpoint, fold_preprocessor, f'{model_name}_fold_{fold_n}')

            oof_probs[model_name] = mlpemb_oof_probs
            experiment_results = pd.DataFrame(cv_results).T
            experiment_results.index.name = f'fold_{model_name}'
            experiment_results.loc['mean_accuracy', 'val_acc_at_best_epoch'] = mean_accuracy
            cv_scores[model_name] = mean_accuracy

            if cfg.general.show_graphs:
                for n, history in enumerate(full_history, start=1):
                    show_results_plot(history, n)

        if cfg.general.train_dl_on_all_data:
            model, preprocessor, _, checkpoint, _ = train_mlp_emb_fold(
                X_train=dataset.X_train,
                y_train=dataset.y_train,
                **params
            )
            results[model_name] = {'model' : model, 'preprocessor' : preprocessor}

            if cfg.general.save_models:
                save_dl_checkpoint(checkpoint, preprocessor, model_name)

        if cfg.general.show_logs:
            print(experiment_results.to_string())

        if cfg.general.save_logs:
            save_dl_experiment_results(model_name, experiment_results)

    return oof_probs, results, cv_scores