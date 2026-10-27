import torch
from collections.abc import Iterable

def get_optimizer(optimizer_name : str, model_parameters : Iterable[torch.nn.Parameter], lr : float, weight_decay : float=0.0) -> torch.optim.Optimizer:
    """
    Получение настроенного оптимизатора по его названию, параметрам модели и параметрам оптимизатора.
    Args:
        optimizer_name : наименование оптимизатора (варианты: 'adamw', 'sgd', 'rmsprop'; иначе создается adam)
        model_parameters : обучаемые тензоры модели
        lr : learning rate для оптимизатора
        weight_decay : коэффициент регуляризации для оптимизатора
    Returns:
        optimizer : настроенный оптимизатор
    """
    if optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(
            model_parameters,
            lr=lr,
            weight_decay=weight_decay,
        )
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(
            model_parameters,
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.9
        )
    elif optimizer_name == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            model_parameters,
            weight_decay=weight_decay,
            lr=lr,
        )
    else:
        optimizer = torch.optim.Adam(
            model_parameters,
            weight_decay=weight_decay,
            lr=lr,
        )
    return optimizer