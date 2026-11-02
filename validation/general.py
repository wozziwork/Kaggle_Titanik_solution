from config import config as cfg

from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV

def build_scoring() -> dict:
    """
    Создает и возвращает словарь метрик, выбранных в конфиге
    """
    available_scorers = {
        'accuracy': 'accuracy',
        'precision': make_scorer(precision_score, zero_division=0),
        'recall': make_scorer(recall_score, zero_division=0),
        'f1': make_scorer(f1_score, zero_division=0),
        'roc_auc': 'roc_auc',
    }
    return {scorer : available_scorers[scorer] for scorer in cfg.validation.scorers}


def build_cv() -> StratifiedKFold:
    """
    Создает и возвращает объект StratifiedKFold с настройками из конфига
    """
    cv = StratifiedKFold(
        n_splits=cfg.validation.n_splits,
        shuffle=cfg.validation.shuffle,
        random_state=cfg.general.seed if cfg.validation.shuffle else None
    )
    return cv

def build_grid_search(
        model : BaseEstimator,
        param_grid : dict[str,list[int | float | str | bool]],
        scoring,
        cv
    ) -> GridSearchCV:
    """
    Создает объект GridSearchCV, настроенный по переданным аргументам и конфигу. Поиск не происходит
    Args:
         model: модель или пайплайн для валидации
         param_grid: словарь с названиями гиперпараметров и списками значений для перебора
         scoring: словарь с метриками
         cv: схема разбиения данных для кросс-валидации
    Returns:
        Настроенный объект GridSearchCV
    """
    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        refit=cfg.search.refit,
        n_jobs=cfg.search.n_jobs,
        verbose=cfg.search.verbose,
        return_train_score=True
    )
    return search