import copy
from collections import defaultdict

from sklearn.metrics import accuracy_score
from torch import Tensor

from tqdm.auto import tqdm
import torch

from dl_models.mlp import MLP
from dl_models.mlp_emb import MLPEmb


def fit_nn(
        model : MLP | MLPEmb,
        train_loader : torch.utils.data.DataLoader,
        optimizer : torch.optim.Optimizer,
        loss_fn : torch.nn.BCEWithLogitsLoss,
        epochs : int,
        early_stopping_patience : int,
        device : str = 'cpu',
        val_inputs : tuple[Tensor, ...] | None = None,
        y_val : Tensor | None = None,
        ) -> tuple[MLP | MLPEmb, dict, int]:
    """
    Выполняет обучение переданной нейросети с необязательной валидацией.
    При наличии валидации и early stopping восстанавливает лучшее сохраненное состояние по validation_loss
    early_stopping_patience = 0 отключает и early_stopping, и выбор лучшего состояния
    Args:
        model : созданная модель с настроенной архитектурой
        train_loader : загрузчик данных, формат батча (*x_inputs, y)
        optimizer : оптимизатор, связанный с параметрами переданной модели
        loss_fn : лосс-функция
        epochs : максимальное количество эпох обучения
        early_stopping_patience : число эпох подряд без уменьшения лучшего val_loss более чем на 0.0001
        device : наименование устройства, на котором будет проходить обучение
        val_inputs : кортеж тензоров с валидационными данными
        y_val : тензор с правильными ответами для валидационных данных
    Returns:
        model : обученная модель
        history : история train loss, и при наличии валидации, val_loss и accuracy
        best_epoch : индекс выбранной эпохи, начиная с 0; -1 если early_stopping отключен или нет валидации
    """
    if val_inputs is not None and y_val is not None:
        val_inputs = tuple(X.to(device) for X in val_inputs)
        y_val = y_val.to(device)

    history = defaultdict(list)
    min_val_loss = float('inf')
    best_model_state = None
    failed_epochs = 0
    best_epoch = -1

    progress = tqdm(range(epochs), desc='Обучение MLP')
    for epoch in progress:
        model.train()
        train_loss_sum = 0
        for *inputs, y_batch in train_loader:
            inputs = [X.to(device) for X in inputs]
            y_batch = y_batch.to(device)

            logits = model(*inputs)
            loss = loss_fn(logits, y_batch)
            train_loss_sum += loss.item() * y_batch.shape[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        history['train_loss'].append(train_loss_sum / len(train_loader.dataset))

        model.eval()
        if val_inputs is not None and y_val is not None:
            with torch.no_grad():
                val_logits = model(*val_inputs)
                val_probs = torch.sigmoid(val_logits)
                val_preds = (val_probs >= 0.5).int()

                val_loss = loss_fn(val_logits, y_val)
                accuracy = accuracy_score(y_val.cpu().numpy().ravel(), val_preds.cpu().numpy().ravel())

                history['val_loss'].append(val_loss.item())
                history['val_accuracy'].append(accuracy)

                progress.set_postfix(
                    val_loss=val_loss.item(),
                    val_accuracy=accuracy,
                )

                if early_stopping_patience:
                    if val_loss.item() < min_val_loss - 0.0001:
                        failed_epochs = 0
                        min_val_loss = val_loss.item()
                        best_model_state = copy.deepcopy(model.state_dict())
                        best_epoch = epoch
                    else:
                        failed_epochs += 1
                        if failed_epochs >= early_stopping_patience:
                            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history, best_epoch
