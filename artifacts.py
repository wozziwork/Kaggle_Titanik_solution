from omegaconf import OmegaConf
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from config import config as cfg
import joblib
import pandas as pd
import torch


def save_model(model :Pipeline, model_name : str) -> None:
    """
    Сохраняет переданную модель в файл "experiment_name_model_name". Если такой уже есть, то он перезаписывается
    Args:
        model: сохраняемая модель - Pipeline с шагом 'model'
        model_name: наименование модели в конфиге для сохранения параметров
    """
    experiment_model_name = cfg.general.experiment_name + '_' + model_name
    joblib.dump(model,cfg.paths.MODELS_DIR / experiment_model_name)


def save_experiment_log(
        model : Pipeline,
        results : dict | pd.DataFrame,
        model_name : str
    ) -> None:
    """
    Сохраняет лог эксперимента, лог содержит: название эксперимента, сид, параметры модели, результаты эксперимента
    Args:
        model: сохраняемая модель - Pipeline с шагом 'model'
        results: словарь результатов CV или DataFrame со сводкой GridSearchCV
        model_name: наименование модели в конфиге для сохранения параметров
    """
    experiment_model_name = cfg.general.experiment_name + '_' + model_name

    # логируем фактические значения параметров, перечисленных в конфиге модели или ее сетке поиска
    params = model.named_steps['model'].get_params()
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



def load_model(file_name : str) -> Pipeline:
    """
    Загружает модель из файла "file_name" в папке saved_models
    """
    path = cfg.paths.MODELS_DIR / file_name
    return joblib.load(path)

def save_dl_checkpoint(checkpoint : dict, preprocessor : ColumnTransformer, model_name : str) -> None:
    experiment_model_name = cfg.general.experiment_name + '_' + model_name
    model_path = cfg.paths.MODELS_DIR / f'{experiment_model_name}.pt'
    preprocessor_path = cfg.paths.MODELS_DIR / f'{experiment_model_name}_preprocessor.joblib'
    torch.save(checkpoint, model_path)
    joblib.dump(preprocessor, preprocessor_path)


def load_dl_checkpoint(experiment_model_name : str) -> tuple[dict, ColumnTransformer]:
    model_path = cfg.paths.MODELS_DIR / f'{experiment_model_name}.pt'
    preprocessor_path = cfg.paths.MODELS_DIR / f'{experiment_model_name}_preprocessor.joblib'
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
    preprocessor = joblib.load(preprocessor_path)
    return checkpoint, preprocessor

def save_dl_experiment_results(model_name : str, experiment_results : pd.DataFrame) -> None:
    experiment_model_name = cfg.general.experiment_name + '_' + model_name
    model_params = OmegaConf.to_container(cfg.models[model_name], resolve=True)
    train_params = OmegaConf.to_container(cfg.dl_train[model_name], resolve=True)

    # Сохранение результатов эксперимента
    # Сохраняются seed, параметры модели и обучения, результаты обучения(CV)
    with (cfg.paths.MODELS_DIR / 'logs.txt').open('a', encoding='utf-8') as file:
        file.write(f'\nExperiment_name: {experiment_model_name}\n')
        file.write(f'Seed: {cfg.general.seed}\n')
        file.write(f'Model_params: {model_params}\n')
        file.write(f'train_params: {train_params}\n')
        file.write(f'Results:\n{pd.DataFrame(experiment_results).to_string()}\n')


