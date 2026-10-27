import pandas as pd
from config import config as cfg
import numpy as np

class Dataset:
    """
    Загрузка train/test данных, а также их подготовка
    """
    def __init__(self):
        self.train = pd.read_csv(cfg.paths.TRAIN_PATH)
        self.test = pd.read_csv(cfg.paths.TEST_PATH)
        self.test_passenger_ids = self.test['PassengerId']

        self.X_train, self.X_test, self.y_train = self._prepare_data()


    def _prepare_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """
        Подготовка данных: отделяет целевую переменную и удаляет столбцы из конфига
        Также создает новый столбец Title, содержащий обращения для пассажиров (Mr, Miss и др)
        Returns: X_train, X_test, y_train
        """
        dropped_cols = cfg.dataset.dropped_columns

        # Достаем из признака Name обращение к человеку (пример: Mr, Miss, Mrs и тд)
        self.train['Title'] = self.train['Name'].str.split(',', n=1, regex=False).str[1].str.split('.', n=1, regex=False).str[0].str.strip()
        self.test['Title'] = self.test['Name'].str.split(',', n=1, regex=False).str[1].str.split('.', n=1, regex=False).str[0].str.strip()

        y_train = self.train['Survived']
        X_train = self.train.drop(columns=['Survived'] + cfg.dataset.dropped_columns)
        X_test = self.test.drop(columns=dropped_cols)


        return X_train, X_test, y_train