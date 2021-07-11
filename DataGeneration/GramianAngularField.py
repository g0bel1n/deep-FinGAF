import numpy as np
import pandas

def normalize(X):
    maxi, mini = max(X), min(X)
    return (2 * X - mini - maxi) / (maxi - mini)


def make_gaf(x: np.ndarray) -> np.ndarray:
    i = np.ones((x.shape[0], 1))
    x = np.reshape(x, (x.shape[0], 1))
    sqrt_val = np.sqrt(i - x ** 2)
    return np.matmul(x, x.T) - np.matmul(sqrt_val, sqrt_val.T)


def polar_encoding(x: np.ndarray, time_stamp: np.ndarray) -> np.ndarray:
    # phi = arcos(x) => r*cos(phi) = r*x
    return time_stamp * x


def fit_transform(df: pandas.DataFrame) -> np.ndarray:
    x = df.to_numpy()
    time_stamp = np.arange(x.shape[0]) / x.shape[0]
    x_norm = normalize(x)
    x_polar = polar_encoding(x_norm, time_stamp)
    return make_gaf(x_polar)
