
import pandas as pd
from sklearn.compose import ColumnTransformer
from torch.utils.data import TensorDataset, DataLoader


from config import config as cfg
from dl_models.mlp_emb import MLPEmb
from optimizer import get_optimizer
from preprocessing import embedding_preprocessor
import torch

from train.fit_nn import fit_nn


def train_mlp_emb_fold(
        X_train : pd.DataFrame,
        y_train : pd.Series,
        batch_size : int,
        hidden_dims : list[int],
        embedding_dims : list[int],
        batch_norm : bool,
        dropout : float,
        lr : float,
        epochs : int,
        early_stopping_patience : int,
        num_workers : int,
        weight_decay : float = 0.,
        optimizer_name : str = 'adam',
        activation_func : str = 'relu',
        X_val : pd.DataFrame | None = None,
        y_val : pd.DataFrame | None = None,
        ) -> tuple[MLPEmb, ColumnTransformer, dict, dict, int]:
    """
    Функция обучения и валидации MLPEmb модели c эмбеддинг слоем для категориальных фичей.
    Подготавливает данные и обучает MLPEmb на переданном разбиении с необязательной валидацией.
    Валидация проводится только при переданных X_val, y_val.
    При включенном early stopping и наличии валидации восстанавливает лучшее сохраненное состояние по validation loss с учетом
    минимального улучшения 0.0001.
    Args:
        X_train : данные для обучения
        y_train : ответы к данным для обучения
        batch_size : размер батча
        hidden_dims : список размеров скрытых слоев MLP
        embedding_dims : список размерностей embedding-векторов для каждого категориального признака
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
        best_epoch : индекс выбранной эпохи, начиная с 0; -1 если не было валидации или ранней остановки
    """

    device = cfg.general.device
    torch.manual_seed(cfg.general.seed)
    generator = torch.Generator().manual_seed(cfg.general.seed)

    preprocessor = embedding_preprocessor()

    X_processed = preprocessor.fit_transform(X_train).astype('float32')
    n_num = len(cfg.dataset.numeric_columns)

    X_num = torch.from_numpy(X_processed[:, :n_num].astype('float32'))
    X_cat = torch.from_numpy(X_processed[:, n_num:].astype('int64') + 1)
    y_train = torch.from_numpy(y_train.to_numpy(dtype='float32')).reshape(-1, 1)

    if X_val is not None and y_val is not None:
        X_val_processed = preprocessor.transform(X_val).astype('float32')
        X_val_num = torch.from_numpy(X_val_processed[:, :n_num].astype('float32')).to(device)
        X_val_cat = torch.from_numpy(X_val_processed[:, n_num:].astype('int64') + 1).to(device)
        X_val = (X_val_num, X_val_cat)
        y_val = torch.from_numpy(y_val.to_numpy(dtype='float32')).reshape(-1, 1).to(device)

    train_dataset = TensorDataset(X_num, X_cat, y_train)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator
    )

    encoder = preprocessor.named_transformers_['categorical'].named_steps['encoder']
    categorical_size = [
        len(categories) + 1 for categories in encoder.categories_
    ]

    mlp_emb = MLPEmb(
        num_numeric=n_num,
        category_size=categorical_size,
        embedding_dims=embedding_dims,
        hidden_dims=hidden_dims,
        batch_norm=batch_norm,
        dropout=dropout,
        activation_func=activation_func,
    )

    mlp_emb.to(device)

    loss_fn = torch.nn.BCEWithLogitsLoss()

    optimizer = get_optimizer(optimizer_name, mlp_emb.parameters(), lr, weight_decay)

    model, history, best_epoch = fit_nn(mlp_emb, train_loader, optimizer, loss_fn, epochs, early_stopping_patience, device, X_val, y_val)

    checkpoint = {
        'model_state_dict': mlp_emb.state_dict(),
        'model_config' : {
            'num_numeric': n_num,
            'category_size': categorical_size,
            'embedding_dims': embedding_dims,
            'hidden_dims': hidden_dims,
            'batch_norm': batch_norm,
            'dropout': dropout,
            'activation_func': activation_func,
        },
    }

    return mlp_emb, preprocessor, history, checkpoint, best_epoch