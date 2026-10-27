import torch
from torch import nn, Tensor


class MLPEmb(nn.Module):
    """
    Класс MLP для бинарной классификации с отдельным embedding-слоем для каждого категориального признака
    и одним выходным логитом.
    Args:
        num_numeric : кол-во численных признаков
        category_size : список с количеством категорий для каждого категориального признака (с учетом категории 'Undefined' = 0)
        embedding_dims : список размерностей embedding-векторов для каждого категориального признака.
            Порядок должен совпадать с category_size
        hidden_dims : список размеров скрытых слоев
        batch_norm : флаг, отвечающий за добавление BatchNorm1d слоев
        dropout : вероятность обнуления активации; при 0 слой не добавляется
        activation_func : наименование функции активации (варианты: gelu, silu, leacky_relu, иначе будет relu)
    """
    def __init__(self,
                 num_numeric : int,
                 category_size : list[int],
                 embedding_dims : list[int],
                 hidden_dims : list[int],
                 batch_norm : bool,
                 dropout : float,
                 activation_func : str | None = None
                 ) -> None:
        super(MLPEmb, self).__init__()

        # создание таблицы с эмбеддингами под каждый категориальный признак
        self.embeddings = nn.ModuleList(
            nn.Embedding(size, dim) for size, dim in zip(category_size, embedding_dims)
        )
        input_dim = num_numeric + sum(embedding_dims)
        layers = []
        dims  = [input_dim, *hidden_dims, 1]


        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i != len(dims) - 2:
                if batch_norm:
                    layers.append(nn.BatchNorm1d(dims[i + 1]))

                if activation_func == 'gelu':
                    layers.append(nn.GELU())
                elif activation_func == 'silu':
                    layers.append(nn.SiLU())
                elif activation_func == 'leacky_relu':
                    layers.append(nn.LeakyReLU(negative_slope=0.1))
                else:
                    layers.append(nn.ReLU())

                if dropout:
                    layers.append(nn.Dropout(p=dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, X_num : Tensor, X_cat : Tensor) -> Tensor:
        # Получаем значения из таблицы эмбеддингов для каждого категориального признака.
        # Затем соединяем с числовыми признаками
        embedded = []
        for i, embedding in enumerate(self.embeddings):
            feature_ind = X_cat[:, i]
            feature_vec = embedding(feature_ind)
            embedded.append(feature_vec)
        X = torch.cat([X_num, *embedded], dim=1)
        return self.network(X)