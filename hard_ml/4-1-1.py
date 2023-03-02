import numpy as np
import pandas as pd
from scipy.stats import ttest_ind


def estimate_first_type_error(
    df_pilot_group,
    df_control_group,
    metric_name,
    alpha=0.05,
    n_iter=10000,
    seed=None
):
    """Оцениваем ошибку первого рода.

    Бутстрепим выборки из пилотной и контрольной групп тех же размеров, считаем долю случаев с значимыми отличиями.
    
    df_pilot_group - pd.DataFrame, датафрейм с данными пилотной группы
    df_control_group - pd.DataFrame, датафрейм с данными контрольной группы
    metric_name - str, названия столбца с метрикой
    alpha - float, уровень значимости для статтеста
    n_iter - int, кол-во итераций бутстрапа
    seed - int or None, состояние генератора случайных чисел.

    return - float, ошибка первого рода
    """
    np.random.seed(seed)
    size = len(df_pilot_group)
    # get data
    pilot = df_pilot_group[metric_name].tolist()
    control = df_control_group[metric_name].tolist()
    # bootstrap
    qnt = 0
    for iter in range(n_iter):
        bootstrap_pilot = np.random.choice(pilot, size)
        bootstrap_control = np.random.choice(control, size)
        pvalue = ttest_ind(bootstrap_pilot, bootstrap_control).pvalue
        if pvalue < alpha:
            qnt += 1
    return qnt / n_iter
