import numpy as np
import pandas as pd
import torch

from omegaconf import OmegaConf
from sklearn.pipeline import Pipeline

from artifacts import load_model, save_predictions, load_dl_checkpoint
from dataset import Dataset
from dl_models.mlp import MLP
from dl_models.mlp_emb import MLPEmb
from preprocessing import prepare_embedding_inputs
from config import config as cfg




def run_inference(dataset : Dataset) -> None:
    """
    Выполняет инференс моделей и ансамблей, включенных в cfg.to_inference.
    Загружает модели и препроцессоры согласно конфигу, получает предсказания
        для dataset.X_test и сохраняет CSV с PassengerId и Survived
    """
    X_test = dataset.X_test
    passenger_ids = dataset.test_passenger_ids
    to_inference = OmegaConf.to_container(cfg.to_inference, resolve=True)

    classic_models = ['logistic_regression', 'knn', 'decision_tree', 'random_forest', 'catboost', 'xgboost', 'lgbm']
    mlp_models = ['mlp', 'mlp_emb']
    device = cfg.general.device

    for model_name, flag in to_inference.items():
        if flag and model_name in classic_models:
            model : Pipeline = load_model(cfg.inference[model_name])
            test_preds = model.predict(X_test)
            save_predictions(passenger_ids, test_preds, model_name)

        if flag and model_name in mlp_models:
            file_name = cfg.inference[model_name]
            checkpoint, preprocessor = load_dl_checkpoint(file_name)
            X_processed = preprocessor.transform(X_test)
            if model_name == 'mlp':
                X = (torch.from_numpy(X_processed.astype('float32')).to(device),)
                mlp_model = MLP(**checkpoint['model_config'])
            else:
                X = prepare_embedding_inputs(X_processed, device)
                mlp_model = MLPEmb(**checkpoint['model_config'])

            mlp_model.load_state_dict(checkpoint['model_state_dict'])
            mlp_model.to(device)
            mlp_model.eval()

            with torch.no_grad():
                logits = mlp_model(*X)
                test_preds = (torch.sigmoid(logits) >= 0.5).int().to('cpu').numpy().ravel()
            save_predictions(passenger_ids, test_preds, model_name)

        if flag and model_name == 'mlp_emb_fold_ensemble':
            probs_sum = np.zeros(len(passenger_ids), dtype='float32')
            for fold_n in range(1, 6):
                checkpoint_name = f'{cfg.inference.mlp_emb_fold_ensemble}_fold_{fold_n}'
                checkpoint, preprocessor = load_dl_checkpoint(checkpoint_name)
                X_processed = preprocessor.transform(X_test)
                X_num, X_cat = prepare_embedding_inputs(X_processed, device)
                fold_mlp_emb = MLPEmb(**checkpoint['model_config'])
                fold_mlp_emb.load_state_dict(checkpoint['model_state_dict'])
                fold_mlp_emb.to(device)
                fold_mlp_emb.eval()
                with torch.no_grad():
                    logits = fold_mlp_emb(X_num, X_cat)
                    probs = torch.sigmoid(logits).to('cpu').numpy().ravel()
                    probs_sum = probs_sum + probs
            fold_ensemble_preds = np.asarray(probs_sum / 5 >= 0.5, dtype = int)
            save_predictions(passenger_ids, fold_ensemble_preds, 'mlp_emb_fold_ensemble')

        if flag and model_name in ['averaging', 'stacking', 'voting']:
            catboost = (load_model(cfg.inference.catboost))
            catboost_preds = catboost.predict(X_test).ravel().astype(int)
            lgbm = load_model(cfg.inference.lgbm)
            lgbm_preds = lgbm.predict(X_test).ravel().astype(int)

            checkpoint, preprocessor = load_dl_checkpoint(cfg.inference.mlp_emb)
            X_processed = preprocessor.transform(X_test)
            X_num, X_cat = prepare_embedding_inputs(X_processed, device)
            mlp_emb = MLPEmb(**checkpoint['model_config'])
            mlp_emb.load_state_dict(checkpoint['model_state_dict'])
            mlp_emb.to(device)
            mlp_emb.eval()
            with torch.no_grad():
                logits = mlp_emb(X_num, X_cat)
                mlp_emb_preds = (torch.sigmoid(logits) >= 0.5).int().to('cpu').numpy().ravel().astype(int)

            catboost_probs = catboost.predict_proba(X_test)[:, 1]
            lgbm_probs = lgbm.predict_proba(X_test)[:, 1]
            mlp_emb_probs = torch.sigmoid(logits).to('cpu').numpy().ravel()

            if model_name == 'voting':
                voting_preds = (catboost_preds + lgbm_preds + mlp_emb_preds >= 2).astype(int)
                save_predictions(passenger_ids, voting_preds, 'voting')

            if model_name == 'averaging':
                average_probs = (catboost_probs + lgbm_probs + mlp_emb_probs) / 3
                average_preds = (average_probs >= 0.5).astype(int)
                save_predictions(passenger_ids, average_preds, 'averaging')

            if model_name == 'stacking':
                meta_model = load_model(cfg.inference.stacking.ridge)
                X_meta = pd.DataFrame({
                    'catboost': catboost_probs,
                    'lgbm': lgbm_probs,
                    'mlp_emb': mlp_emb_probs,
                })
                stacking_preds = (meta_model.predict(X_meta) >= 0.5).astype(int)
                save_predictions(passenger_ids, stacking_preds, 'stacking')

