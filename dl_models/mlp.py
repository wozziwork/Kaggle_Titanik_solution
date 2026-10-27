from torch import nn, Tensor


class MLP(nn.Module):
    """
    Класс MLP модели для бинарной классификации с одним выходным логитом
    Args:
        input_dim : число входных признаков
        hidden_dims : список размеров скрытых слоев
        batch_norm : флаг, отвечающий за добавление BatchNorm1d слоев
        dropout : вероятность обнуления активации; при 0 слой не добавляется
        activation_func : наименование функции активации (варианты: gelu, silu, leacky_relu, иначе будет relu)
    """
    def __init__(self,
                 input_dim : int,
                 hidden_dims : list[int],
                 batch_norm : bool,
                 dropout : float,
                 activation_func : str | None = None
                 ) -> None:
        super(MLP, self).__init__()
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

    def forward(self, X : Tensor) -> Tensor:
        X = self.network(X)
        return X