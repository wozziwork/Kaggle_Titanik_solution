

import pandas as pd
from sklearn.compose import ColumnTransformer

from torch.utils.data import TensorDataset, DataLoader

from dl_models.mlp import MLP

from config import config as cfg
from optimizer import get_optimizer
from preprocessing import build_preprocessor
import torch

from train.fit_nn import fit_nn


def train_mlp_fold(
        X_train : pd.DataFrame,
        y_train : pd.Series,
        batch_size : int,
        hidden_dims : list[int],
        batch_norm : bool,
        dropout : float,
        lr : float,
        epochs : int,
        early_stopping_patience : int,
        num_workers : int,
        weight_decay: float = 0.,
        optimizer_name : str = 'adam',
        activation_func : str = 'relu',
        X_val : pd.DataFrame | None = None,
        y_val : pd.DataFrame | None = None,
        ) -> tuple[MLP, ColumnTransformer, dict, dict, int]:
    """
    Функция обучения и валидации MLP модели с переданными параметрами.
    Подготавливает данные и обучает MLP на переданном разбиении с необязательной валидацией.
    Валидация проводится только при переданных X_val, y_val.
    При включенном early stopping и наличии валидации восстанавливает лучшее сохраненное состояние по validation loss с учетом
    минимального улучшения 0.0001.
    Args:
        X_train : данные для обучения
        y_train : ответы к данным для обучения
        batch_size : размер батча
        hidden_dims : список размеров скрытых слоев MLP
        batch_norm : флаг, отвечающий за добавление BatchNorm1d слоев
        dropout : вероятность обнуления активации; при 0 слой не добавляется
        lr : шаг обучения для оптимизатора
        epochs : максимальное количество эпох обучения
        early_stopping_patience : количество нерезультативных эпох подряд перед остановкой обучения
        num_workers : количество процессов для загрузки данных
        weight_decay : коэффициент регуляризации для оптимизатора
        optimizer_name : наименование оптимизатора (варианты: 'adamw', 'sgd', 'rmsprop'; иначе создается adam)
        activation_func : наименование функции активации (варианты: gelu, silu, leacky_relu, иначе будет relu)
        X_val : данные для валидации. По умолчанию отсутствуют
        y_val : ответы к данным для валидации. По умолчанию отсутствуют
    Returns:
        mlp : обученная модель
        preprocessor : обученный препроцессор
        history : история train loss, и при наличии валидации, val_loss и accuracy
        checkpoint : словарь с состоянием модели и настройками архитектуры
        best_epoch : индекс выбранной эпохи, начиная с 0; -1 если ранней остановки не было
    """

    device = cfg.general.device
    torch.manual_seed(cfg.general.seed)
    generator = torch.Generator().manual_seed(cfg.general.seed)

    preprocessor = build_preprocessor(sparse_output=False)

    X_train = preprocessor.fit_transform(X_train, y_train).astype('float32')
    X_train = torch.from_numpy(X_train)
    y_train = torch.from_numpy(y_train.to_numpy(dtype='float32')).reshape(-1, 1)

    if X_val is not None and y_val is not None:
        X_val = preprocessor.transform(X_val).astype('float32')
        X_val = (torch.from_numpy(X_val).to(device), ) # для нормальной работы с fit_nn
        y_val = torch.from_numpy(y_val.to_numpy(dtype='float32')).reshape(-1, 1).to(device)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator
    )

    input_dim = X_train.shape[1]
    mlp = MLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        batch_norm=batch_norm,
        dropout=dropout,
        activation_func=activation_func,
    )
    mlp.to(device)

    loss_fn = torch.nn.BCEWithLogitsLoss()

    optimizer = get_optimizer(optimizer_name, mlp.parameters(), lr, weight_decay)

    mlp, history, best_epoch = fit_nn(mlp, train_loader, optimizer, loss_fn, epochs, early_stopping_patience, device, X_val, y_val)

    checkpoint = {
        'model_state_dict': mlp.state_dict(),
        'model_config' : {
            'input_dim': input_dim,
            'hidden_dims': hidden_dims,
            'batch_norm': batch_norm,
            'dropout': dropout,
            'activation_func': activation_func,
        },
    }

    return mlp, preprocessor, history, checkpoint, best_epoch