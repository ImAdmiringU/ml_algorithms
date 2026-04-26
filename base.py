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

    res = -np.sum([(i / len(y)) * np.log2(i / len(y)) for i in np.bincount(y) if i != 0])

    return res

def gini(y: np.array) -> float:
    pass

def information_gain(main_node_impurity, y_left, y_right):
    '''
    Функция для расчета прироста
    информации (IG) после сплита

    Параметры
    ---------
    main_node_impurity : float
        Значение энтропии в исходном узле
    y_left : np.array
        Классы объектов после сплита для
        левой ноды
    y_right : np.array
        Классы объектов после сплита для
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
           - len(y_left) / y_amount * entropy(y_left)
           - len(y_right) / y_amount * entropy(y_right))
    
    return res
