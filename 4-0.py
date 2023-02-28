import numpy as np

def get_bernoulli_confidence_interval(values: np.array):
    """Вычисляет доверительный интервал для параметра распределения Бернулли.

    :param values: массив элементов из нулей и единиц.
    :return (left_bound, right_bound): границы доверительного интервала.
    """
    # YOUR_CODE_HERE
    z = 1.96
    p = sum(values) / len(values)
    left_bound = max(0, p - z * ((p * (1 - p)/len(values)) ** 0.5))
    right_bound = min(1, p + z * ((p * (1 - p)/len(values)) ** 0.5))
    return left_bound, right_bound
