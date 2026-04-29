import numpy as np
import pandas as pd

def entropy(y: np.array) -> float:
    '''
    Функция для расчета энтропии Шеннона

    Параметры
    ---------
    y : np.array
        Одномерный numpy вектор классов

    Возвращаемое значение
    ---------------------
    res : float
        Энтропия в текущем узле    
    '''

    res = -np.sum([(i / len(y)) * np.log2(i / len(y)) for i in np.bincount(y.astype(int)) if i != 0])

    return res

def gini(y: np.array) -> float:
    pass

def mse(y: np.array) -> float:
    '''
    Функция MSE (Mean Squared Error)

    Параметры
    ---------
    y : np.array
        Одномерный numpy вектор
        значений наблюдений

    Возвращаемое значение
    ---------------------
    res : float
        MSE в узле    
    '''

    res = np.mean((y - np.mean(y))**2)

    return res

def mae(y: np.array) -> float:
    pass

def gain(main_node_impurity, y_left, y_right, criterion_func):
    '''
    Функция для расчета прироста
    информации после сплита

    Параметры
    ---------
    main_node_impurity : float
        Значение impurity в исходном узле
    y_left : np.array
        Таргет объектов после сплита для
        левой ноды
    y_right : np.array
        Таргет объектов после сплита для
        правой ноды
            
    Временные переменные
    --------------------
    y_amount : int
        Общее количество объектов в
        исходном узле

    Возвращаемое значение
    ---------------------
    res : float
        Прирост информации при
        текущем сплите 
    '''

    y_amount = len(y_left) + len(y_right)

    res = (main_node_impurity
           - (len(y_left) / y_amount * criterion_func(y_left)
           + len(y_right) / y_amount * criterion_func(y_right))
           )
    
    return res
