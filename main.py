import time
import pandas as pd
from omegaconf import OmegaConf

from dataset import Dataset
from train.ensembles import ensemble_experiment
from train.train import train
from config import config as cfg
from train.train_dl import train_dl
from inference import run_inference

if __name__ == "__main__":
    start = time.time()
    dataset = Dataset()

    if cfg.general.train:
        models, oof_probs, classic_scores = train(dataset)
        oof_probs, results, dl_scores = train_dl(dataset, oof_probs)
        ensemble_experiment(dataset, oof_probs)
        cv_scores = {**classic_scores, **dl_scores}
        summary = pd.DataFrame(cv_scores.items(), columns=['model', 'cv_accuracy'])
        print(summary.to_string(index=False))

    if cfg.general.inference:
        run_inference(dataset)
        models_to_inference = []
        for key, value in OmegaConf.to_container(cfg.to_inference, resolve = True).items():
            if value: models_to_inference.append(key)
        print(f'Имя эксперимента : {cfg.general.experiment_name}')
        print(f'Модели для инференса : {models_to_inference}')
        print('Модели лежат в папке saved_models, путь к каждой модели задается через config (раздел inference)')
        print("Полученные предсказания лежат в папке inference_predicts и содержат названия текущего эксперимента")
    end = time.time()
    print(f'Time: {end - start}')



