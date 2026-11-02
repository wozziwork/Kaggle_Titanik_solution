import pandas as pd

from omegaconf import OmegaConf
from config import config as cfg

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score

from dataset import Dataset
from train.train_dl import get_params
from validation.stacking import evaluate_stacking_cv
from artifacts import save_oof_predictions, save_experiment_log, save_model


def voting(oof_res : pd.DataFrame) -> tuple[float, float]:
    """
    Оценка ансамбля голосования и усреднения catboost, mlp_emb, lgbm на их out of fold предсказаниях.
    Возвращает accuracy для ансамблей голосованием и усреднением.
    Args:
        oof_res :датафрейм, содержащий oof предсказания всех моделей ансамбля и правильные ответы
    Return:
        accuracy_averaging : accuracy для ответов, полученных усреднением вероятностей
        accuracy_voting : accuracy для ответов, полученных голосованием
    """
    cat_preds = (oof_res['catboost'] >= 0.5).astype(int)
    lgbm_preds = (oof_res['lgbm'] >= 0.5).astype(int)
    mlp_preds = (oof_res['mlp_emb'] >= 0.5).astype(int)
    voting_preds = ((cat_preds + lgbm_preds + mlp_preds) >= 2).astype(int)

    average_probs = (oof_res['catboost'] + oof_res['lgbm'] + oof_res['mlp_emb']) / 3
    average_preds = (average_probs >= 0.5)

    accuracy_averaging = accuracy_score(oof_res['y_true'], average_preds)
    accuracy_voting = accuracy_score(oof_res['y_true'], voting_preds)
    return accuracy_averaging, accuracy_voting

def stacking_cv(meta_model : Ridge, dataset : Dataset) -> float:
    """
    Оценка стекинга catboost, lgbm, mlpemb (настройки модели берутся из конфига).
    Возвращает полученную на внешних валидационных фолдах accuracy.
    """
    # создаем модели, которые будут обучаться для стекинга
    lgbm = (LGBMClassifier(**OmegaConf.to_container(cfg.models.lgbm, resolve=True))
            .set_params(n_estimators=cfg.models.stacking.lgbm_n_estimators ,early_stopping_round=0))
    catboost = (CatBoostClassifier(**OmegaConf.to_container(cfg.models.catboost, resolve=True))
            .set_params(iterations=cfg.models.stacking.catboost_iterations ,early_stopping_rounds=None, use_best_model=False))
    mlp_emb_params = get_params('mlp_emb')
    stacking_preds, accuracy = evaluate_stacking_cv(
        dataset.X_train, dataset.y_train,
        catboost, lgbm, mlp_emb_params, meta_model
    )
    return accuracy

def stacking_train(meta_model : Ridge, oof_results : pd.DataFrame) -> Ridge:
    """
    Обучает Ridge на всей OOF-таблице
    """
    y_true = oof_results['y_true']
    X = oof_results[['catboost', 'lgbm', 'mlp_emb']]
    meta_model.fit(X, y_true)
    return meta_model


def ensemble_experiment(dataset : Dataset, oof_probs : dict) -> None:
    """
    Эксперимент-оценка ансамблей(stacking с Ridge, voting, averaging) на предсказанных базовыми моделями oof вероятностях.
    Оценка стекинга выполняется отдельно через вложенную кросс-валидацию.
    Также обучает Ridge (метамодель) на всей OOF-таблице и сохраняет его согласно конфигу.
    OOF-таблица формируется из словаря oof_probs или читается из файла, путь к которому указан в конфиге.
    Выводит и сохраняет результаты оценок согласно настройкам конфига.
    Args:
        dataset : объект Dataset для получения столбцов PassengerId, y_true и CV стекинга
        oof_probs :словарь, содержащий предсказанные базовыми моделями oof вероятности (если они обучались)
    """
    if cfg.to_train.voting or cfg.to_train.stacking or cfg.to_train.averaging:
        # Если 3 базовые модели ансамбля обучаются, то
        # согласно конфигу происходит формирование и сохранение таблицы или только формирование (без сохранения)
        if all([cfg.to_train.catboost, cfg.to_train.lgbm, cfg.to_train.mlp_emb]):
            if cfg.general.save_oof:
                oof_res = save_oof_predictions(dataset, oof_probs, cfg.paths.OOF_PROBS)
            else:
                oof_res = save_oof_predictions(dataset, oof_probs, None)
        else:
            # Если 3 базовых модели в данном эксперименте не обучаются, то читаем oof предсказания по пути из конфига
            oof_res = pd.read_csv(cfg.paths.OOF_PROBS)

        if cfg.to_train.voting or cfg.to_train.averaging:
            accuracy_averaging, accuracy_voting = voting(oof_res)
            results = pd.DataFrame([{'accuracy_averaging': accuracy_averaging, 'accuracy_voting': accuracy_voting}])
            if cfg.general.show_logs:
                print(results)
            if cfg.general.save_logs:
                save_experiment_log(model=None, results=results, model_name='voting')

        if cfg.to_train.stacking:
            alpha = cfg.models.stacking.alpha
            meta_model = Ridge(alpha=alpha)
            if cfg.models.stacking.cv:
                accuracy = stacking_cv(meta_model, dataset)
                results = pd.DataFrame([{'Stacking accuracy': accuracy, 'Ridge alpha': alpha}])

                if cfg.general.show_logs:
                    print(f'Accuracy: {accuracy}')
                    print(f'Ridge alpha = {alpha}')
                if cfg.general.save_logs:
                    save_experiment_log(model=meta_model, results=results, model_name='stacking')

            meta_model = stacking_train(meta_model, oof_res)

            if cfg.general.save_models:
                save_model(meta_model, 'meta_model_ridge')