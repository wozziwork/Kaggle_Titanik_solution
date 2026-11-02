import numpy as np
import torch
import pandas as pd
import copy

from config import config as cfg

from train.train_mlp_emb import train_mlp_emb_fold

from validation.general import build_cv
from validation.boosting import get_boosting_oof
from validation.neural import mlp_cv

from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from preprocessing import build_catboost_preprocessor, build_category_caster_preprocessor, prepare_embedding_inputs


def evaluate_stacking_cv(
        X : pd.DataFrame,
        y : pd.Series,
        catboost : CatBoostClassifier,
        lgbm : LGBMClassifier,
        mlp_emb_params : dict,
        meta_model : Ridge,
    ) -> tuple[np.ndarray, float]:
    """
    Обучение и валидация стекинга из catboost, lgbm, MLPEmb.
    Для каждого внешнего фолда:
        1) На его обучающей части проводится внутренняя кросс-валидация и
            полученные oof вероятности используются для обучения мета-модели.
        2) Новые базовые модели обучаются на всей обучающей части внешнего фолда
            и получают вероятности для внешней валидационной части.
        3) Обученная метамодель преобразует их в итоговые предсказания.
    Предсказания всех внешних фолдов собираются в массив в исходном порядке, после чего считается accuracy
    Args:
        X : данные для оценки стекинга
        y : ответы к данным
        catboost : настроенная необученная catboost-модель (early_stopping отключен, количество деревьев фиксировано)
        lgbm : настроенная lgbm-модель (early_stopping отключен, количество деревьев фиксировано)
        mlp_emb_params : параметры MLPEmb и его обучения
        meta_model : необученная мета-модель с настроенной архитектурой (не изменяется в ходе cv)
    Returns:
        stacking_preds : массив oof предсказаний (порядок сохранен)
        accuracy : подсчитанная accuracy для всех oof-предсказаний стекинга
    """
    outer_cv = build_cv()
    stacking_preds = np.full(len(y), np.nan)

    for train_outer_idx, val_outer_idx in outer_cv.split(X, y):
        # для обучения базовых моделей, получения их oof предсказаний и обучения стекинга
        X_train, y_train = X.iloc[train_outer_idx], y.iloc[train_outer_idx]
        X_val, y_val = X.iloc[val_outer_idx], y.iloc[val_outer_idx] # для валидации стекинга

        #обучение стекинга на ооф предсказаниях
        _, _, _, _ ,mlp_emb_oof = mlp_cv(X_train, y_train, 'mlp_emb', **mlp_emb_params)
        catboost_oof = get_boosting_oof(catboost, X_train, y_train)
        lgbm_oof = get_boosting_oof(lgbm, X_train, y_train)
        # записываем ооф предсказания
        X_meta = pd.DataFrame({
            'catboost' : catboost_oof,
            'lgbm' : lgbm_oof,
            'mlp_emb' : mlp_emb_oof,
        })

        ridge = copy.deepcopy(meta_model)
        ridge.fit(X_meta, y_train)

        # обучение всех моделей и получение по их предиктам предсказания стекинга
        catboost_val_preprocessor = build_catboost_preprocessor()
        catboost_val = copy.deepcopy(catboost)

        lgbm_val_preprocessor = build_category_caster_preprocessor()
        lgbm_val = copy.deepcopy(lgbm)

        X_train_lgbm = lgbm_val_preprocessor.fit_transform(X_train, y_train)
        X_train_catboost = catboost_val_preprocessor.fit_transform(X_train, y_train)

        lgbm_val.fit(X_train_lgbm, y_train)
        catboost_val.fit(X_train_catboost, y_train)

        mlp_emb_val, mlp_emb_val_preprocessor, _, _, _ = train_mlp_emb_fold(X_train, y_train, **mlp_emb_params)

        X_val_lgbm = lgbm_val_preprocessor.transform(X_val)
        X_val_catboost = catboost_val_preprocessor.transform(X_val)
        X_val_mlpemb = mlp_emb_val_preprocessor.transform(X_val)

        device = cfg.general.device
        X_num, X_cat = prepare_embedding_inputs(X_val_mlpemb, device)
        mlp_emb_val.eval()
        with torch.no_grad():
            logits = mlp_emb_val(X_num, X_cat)
            mlp_emb_val_probs = torch.sigmoid(logits).cpu().numpy().ravel()

        X_val_meta = pd.DataFrame({
            'catboost' : catboost_val.predict_proba(X_val_catboost)[:, 1],
            'lgbm' : lgbm_val.predict_proba(X_val_lgbm)[:, 1],
            'mlp_emb' : mlp_emb_val_probs,
        })
        predicts = (ridge.predict(X_val_meta) >= 0.5)
        stacking_preds[val_outer_idx] = predicts

    return stacking_preds, accuracy_score(y, stacking_preds)



