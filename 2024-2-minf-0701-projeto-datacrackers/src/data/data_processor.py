import pandas as pd
from sklearn.utils import shuffle


class DataProcessor:
    @staticmethod
    def shuffle_data(data: pd.DataFrame, random_state: int = 42):
        return shuffle(data, random_state=random_state)

    @staticmethod
    def split_data(data: pd.DataFrame, train_ratio: float = 0.6, validation_ratio: float = 0.2):
        train_size = int(train_ratio * len(data))
        validation_size = int(validation_ratio * len(data))

        train_data = data.iloc[:train_size]
        validation_data = data.iloc[train_size:train_size + validation_size]
        test_data = data.iloc[train_size + validation_size:]

        return {
            'train': train_data,
            'validation': validation_data,
            'test': test_data
        }

    @staticmethod
    def calculate_correlation(data: pd.DataFrame, target_column: str):
        return data.corr()[target_column].drop(target_column)
