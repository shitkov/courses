import numpy as np
import pandas as pd
from scipy.stats import ttest_ind


def estimate_second_type_error(
    df_pilot_group,
    df_control_group,
    metric_name,
    effects,
    alpha=0.05,
    n_iter=10000,
    seed=None
):
    """Оцениваем ошибки второго рода.

    Бутстрепим выборки из пилотной и контрольной групп тех же размеров, добавляем эффект к пилотной группе,
    считаем долю случаев без значимых отличий.
    
    df_pilot_group - pd.DataFrame, датафрейм с данными пилотной группы
    df_control_group - pd.DataFrame, датафрейм с данными контрольной группы
    metric_name - str, названия столбца с метрикой
    effects - List[float], список размеров эффектов ([1.03] - увеличение на 3%).
    alpha - float, уровень значимости для статтеста
    n_iter - int, кол-во итераций бутстрапа
    seed - int or None, состояние генератора случайных чисел

    return - dict, {размер_эффекта: ошибка_второго_рода}
    """
    np.random.seed(seed)
    size = len(df_pilot_group)
    # get data
    pilot = df_pilot_group[metric_name].tolist()
    control = df_control_group[metric_name].tolist()
    # bootstrap
    ans_dict = {}
    for effect in effects:
        qnt = 0
        for iter in range(n_iter):
            bootstrap_pilot = np.random.choice(pilot, size) * effect
            bootstrap_control = np.random.choice(control, size)
            pvalue = ttest_ind(bootstrap_pilot, bootstrap_control).pvalue
            if pvalue >= alpha:
                qnt += 1
        ans_dict[effect] = qnt / n_iter
    return ans_dict