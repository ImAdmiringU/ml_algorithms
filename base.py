import numpy as np
import pandas as pd

def entropy(y_data: np.array) -> float:
    '''
    Функция для расчета энтропии Шеннона

    Параметры
    ---------
    y_data : np.array
        Одномерный numpy вектор классов (0, 1)
            
    Временные переменные
    --------------------
    pi_left : float
        Отношение (вероятность) положительного
        класса (1), ко всем объектам вектора

    Возвращаемое значение
    ---------------------
    res : float
        Энтропия в текущем узле    
    '''

    pi_left = np.sum(y_data) / len(y_data)

    # Если класс разделен так, что все
    # элементы состоят из одного класса
    if pi_left == 0 or pi_left == 1:
        return 0.0

    res = -((pi_left * np.log2(pi_left)) + ((1 - pi_left) * np.log2(1 - pi_left)))

    return res

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
