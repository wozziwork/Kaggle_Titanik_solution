from pathlib import Path

from omegaconf import OmegaConf
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from config import config as cfg
import joblib
import pandas as pd
import torch

from dataset import Dataset



def save_model(model : BaseEstimator, model_name : str) -> None:
    """
    Сохраняет переданную модель или пайплайн в файл "experiment_name_model_name". Если такой уже есть, то он перезаписывается
    Args:
        model: сохраняемая модель или Pipeline с шагом 'model'
        model_name: наименование модели
    """
    experiment_model_name = cfg.general.experiment_name + '_' + model_name
    joblib.dump(model,cfg.paths.MODELS_DIR / experiment_model_name)


def save_experiment_log(
        model : BaseEstimator | None,
        results : dict | pd.DataFrame,
        model_name : str
    ) -> None:
    """
    Сохраняет лог эксперимента, лог содержит: название эксперимента, сид, параметры модели, результаты эксперимента
    Args:
        model: сохраняемая модель или Pipeline с шагом 'model'
        results: словарь результатов CV или DataFrame со сводкой GridSearchCV
        model_name: наименование модели в конфиге для сохранения параметров
    """
    experiment_model_name = cfg.general.experiment_name + '_' + model_name

    # логируем фактические значения параметров, перечисленных в конфиге модели или ее сетке поиска
    filtered_params = None
    if model is not None:
        if isinstance(model, Pipeline):
            params = model.named_steps['model'].get_params()
        else:
            params = model.get_params()
        param_names = set(cfg.models[model_name])
        param_names.update(name.removeprefix('model__') for name in cfg.param_grids.get(model_name,{}))
        filtered_params = {}
        for name, value in params.items():
            if name in param_names:
                filtered_params[name] = value

    # сохранение результатов эксперимента в виде:
    # Имя эксперимента, Параметры модели, результаты cv/gridsearchCV
    with (cfg.paths.MODELS_DIR / 'logs.txt').open('a', encoding='utf-8') as file:
        file.write(f'\nExperiment_name: {experiment_model_name}\n')
        file.write(f'Seed: {cfg.general.seed}\n')
        file.write(f'Params: {filtered_params}\n')
        file.write(f'Metrics:\n{pd.DataFrame(results).to_string()}\n')



def load_model(file_name : str) -> Pipeline | Ridge:
    """
    Загружает модель из файла "file_name" в папке saved_models
    """
    path = cfg.paths.MODELS_DIR / file_name
    return joblib.load(path)


def save_dl_checkpoint(checkpoint : dict, preprocessor : ColumnTransformer, model_name : str) -> None:
    """
    Сохраняет переданный словарь-чекпоинт и препроцессор в файлы с названиями
        {cfg.general.experiment_name}_{model_name}.pt и
        {cfg.general.experiment_name}_{model_name}_preprocessor.joblib соответственно
        (по пути, указанному в cfg.paths.MODELS_DIR)
    Args:
        checkpoint : словарь с состоянием модели и настройками архитектуры
        preprocessor : обученный препроцессор
        model_name : имя модели
    """
    experiment_model_name = cfg.general.experiment_name + '_' + model_name
    model_path = cfg.paths.MODELS_DIR / f'{experiment_model_name}.pt'
    preprocessor_path = cfg.paths.MODELS_DIR / f'{experiment_model_name}_preprocessor.joblib'
    torch.save(checkpoint, model_path)
    joblib.dump(preprocessor, preprocessor_path)


def load_dl_checkpoint(experiment_model_name : str) -> tuple[dict, ColumnTransformer]:
    """
    Функция загрузки словаря с состоянием модели и настройками архитектуры
    из файла {experiment_model_name}.pt, а также обученный препроцессор
    из файла{experiment_model_name}_preprocessor.joblib.
    Оба файла загружаются из папки, указанной в cfg.paths.MODELS_DIR
    Args:
        experiment_model_name : составное имя эксперимента и модели (без расширения .pt),
            используемое для загрузки чекпоинта и препроцессора
    Returns:
        checkpoint : словарь с состоянием модели и настройками архитектуры
        preprocessor : обученный препроцессор
    """
    model_path = cfg.paths.MODELS_DIR / f'{experiment_model_name}.pt'
    preprocessor_path = cfg.paths.MODELS_DIR / f'{experiment_model_name}_preprocessor.joblib'
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
    preprocessor = joblib.load(preprocessor_path)
    return checkpoint, preprocessor

def save_dl_experiment_results(model_name : str, experiment_results : pd.DataFrame) -> None:
    """
    Сохраняет результаты эксперимента с обучением DL-модели
    Args:
        model_name : имя модели
        experiment_results : датафрейм с результатами эксперимента
    """
    experiment_model_name = cfg.general.experiment_name + '_' + model_name
    model_params = OmegaConf.to_container(cfg.models[model_name], resolve=True)
    train_params = OmegaConf.to_container(cfg.dl_train[model_name], resolve=True)

    # Сохраняются seed, параметры модели и обучения, результаты обучения(CV)
    with (cfg.paths.MODELS_DIR / 'logs.txt').open('a', encoding='utf-8') as file:
        file.write(f'\nExperiment_name: {experiment_model_name}\n')
        file.write(f'Seed: {cfg.general.seed}\n')
        file.write(f'Base_model_params: {model_params}\n')
        file.write(f'train_params: {train_params}\n')
        file.write(f'Results:\n{pd.DataFrame(experiment_results).to_string()}\n')


def save_oof_predictions(dataset : Dataset, oof_probs : dict, path : Path | None = None) -> pd.DataFrame:
    """
    Создает и сохраняет датафрейм по словарю с oof_probs. Путь сохранения передается аргументом.
    Если путь None, то сохранения не происходит.
    Датафрейм содержит столбцы:
        PassengerId
        y_true : выжил пассажир или нет
        catboost, mlp_emb, lgbm : предсказанные каждой из трех моделей вероятности выживания
    Returns:
        res : созданный DataFrame с ответами и OOF-вероятностями моделей
    """
    res = pd.DataFrame({
        'PassengerId': dataset.train['PassengerId'],
        'y_true': dataset.y_train,
        'catboost': oof_probs['catboost'],
        'mlp_emb': oof_probs['mlp_emb'],
        'lgbm': oof_probs['lgbm'],
    })
    if path is not None:
        res.to_csv(path, index=False)
    return res

def save_predictions(passenger_ids, predictions, model_name):
    """
    Сохраняет таблицу предсказаний модели для инференса.
    Таблица содержит столбцы PassengerId, Survived
    """
    file_name = f'predictions_{cfg.general.experiment_name}_{model_name}.csv'
    pd.DataFrame({
        'PassengerId': passenger_ids,
        'Survived': predictions,
    }).to_csv(cfg.paths.INFERENCE / file_name, index=False)


