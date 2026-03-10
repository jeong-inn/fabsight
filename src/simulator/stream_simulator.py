import numpy as np
import pandas as pd
from datetime import datetime

class SensorStreamSimulator:
    def __init__(self, X: pd.DataFrame, y: pd.Series, window_size: int = 20):
        self.X = X.reset_index(drop=True)
        self.y = y.reset_index(drop=True)
        self.window_size = window_size
        self.pointer = 0

    def get_next_window(self):
        end = min(self.pointer + self.window_size, len(self.X))
        X_window = self.X.iloc[self.pointer:end]
        y_window = self.y.iloc[self.pointer:end]
        self.pointer = end if end < len(self.X) else 0
        return X_window, y_window

    def get_random_sample(self, n: int = 20):
        idx = np.random.choice(len(self.X), size=n, replace=False)
        return self.X.iloc[idx], self.y.iloc[idx]
