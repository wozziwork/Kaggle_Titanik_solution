import pandas as pd
from omegaconf import OmegaConf

from artifacts import save_dl_experiment_results
from dataset import Dataset
from visualization import show_results_plot
from train.train import train
from config import config as cfg
import time

from validation import mlp_cv, search_mlp_cv, search_mlp_emb_optuna


def get_params(name):
    return {**OmegaConf.to_container(cfg.models[name], resolve=True), **OmegaConf.to_container(cfg.dl_train[name], resolve=True)}


if __name__ == "__main__":
    start = time.time()
    dataset = Dataset()

    if cfg.general.train:
        models = train(dataset)

        if cfg.to_train.mlp:
            model_name = 'mlp'
            params = get_params(model_name)

            if cfg.to_grid_search.mlp:
                param_grid = OmegaConf.to_container(cfg.param_grids.mlp, resolve=True)
                best_acc, best_params, search_results = search_mlp_cv(dataset,model_name, param_grid, params)
                experiment_results = pd.DataFrame(search_results).T
                experiment_results.loc['best_params'] = None
                experiment_results.at['best_params', 'params'] = best_params
                experiment_results.loc['best_accuracy', 'mean_accuracy'] = best_acc

            else:
                cv_results, full_history, mean_accuracy = mlp_cv(
                    dataset,
                    model_name,
                    **params
                )

                experiment_results = pd.DataFrame(cv_results).T
                experiment_results.index.name = f'fold_{model_name}'
                experiment_results.loc['mean_accuracy', 'val_acc_at_best_loss'] = mean_accuracy

                if cfg.general.show_graphs:
                    for n, history in enumerate(full_history, start=1):
                        show_results_plot(history, n)

            print(experiment_results.to_string())

            if cfg.general.save_logs:
                save_dl_experiment_results(model_name, experiment_results)

        if cfg.to_train.mlp_emb:
            model_name = 'mlp_emb'
            params = get_params(model_name)

            if cfg.to_grid_search[model_name]:
                param_grid = OmegaConf.to_container(cfg.param_grids[model_name], resolve=True)
                best_acc, best_params, search_results = search_mlp_cv(dataset, model_name, param_grid, params)
                experiment_results = pd.DataFrame(search_results).T
                experiment_results.loc['best_params'] = None
                experiment_results.at['best_params', 'params'] = best_params
                experiment_results.loc['best_accuracy', 'mean_accuracy'] = best_acc
            else:
                cv_results, full_history, mean_accuracy = mlp_cv(
                    dataset,
                    model_name,
                    **params
                )
                experiment_results = pd.DataFrame(cv_results).T
                experiment_results.index.name = f'fold_{model_name}'
                experiment_results.loc['mean_accuracy', 'val_acc_at_best_loss'] = mean_accuracy

                if cfg.general.show_graphs:
                    for n, history in enumerate(full_history, start=1):
                        show_results_plot(history, n)

            print(experiment_results.to_string())

            if cfg.general.save_logs:
                save_dl_experiment_results(model_name, experiment_results)


    end = time.time()
    print(f'Time: {end - start}')



