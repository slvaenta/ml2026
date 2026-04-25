import numpy as np



# Вспомогательные функции

def rv(value_list):
    return np.array([value_list])

def cv(value_list):
    return np.transpose(rv(value_list))


# Функции для тестов

def f1(v):
    assert v.shape == (1, 1)
    x = float(v[0,0])
    return (2 * x + 3) ** 2

def df1(v):
    assert v.shape == (1, 1)
    x = float(v[0,0])
    return 2 * 2 * (2 * x + 3)

def f2(v):
    assert v.shape == (2, 1)
    x = float(v[0,0]); y = float(v[1,0])
    return (x - 2.) * (x - 3.) * (x + 3.) * (x + 1.) + (x + y - 1.) ** 2

def df2(v):
    assert v.shape == (2, 1)
    x = float(v[0,0]); y = float(v[1,0])
    return cv([(-3. + x) * (-2. + x) * (1. + x) + (-3. + x) * (-2. + x) * (3. + x) + (-3. + x) * (1. + x) * (3. + x) + (-2. + x) * (1. + x) * (3. + x) + 2 * (-1. + x + y), 2 * (-1. + x + y)])


def super_simple_separable():
    X = np.array([[2, 3, 9, 12],
                  [5, 2, 6, 5]])
    Y = np.array([[1, -1, 1, -1]])
    return X, Y

def separable_medium():
    X = np.array([[2, -1, 1, 1],
                  [-2, 2, 2, -1]])
    Y = np.array([[1, -1, 1, -1]])
    return X, Y

sep_e_separator = np.array([[-0.40338351], [1.1849563]]), np.array([[-2.26910091]])


# Ваше решение идёт тут

# Задание 2

def grad_desc(f, df, x0, eta, T):
    return None

def num_grad(f, delta = 0.001):
    return None

def num_grad_desc(f, x0, eta, T):
    return None


# Задание 3

def hinge(v):
    return None

def hinge_loss(X, Y, th, th0):
    return None

def svm_obj(X, Y, th, th0, lam):
    return None


# Задание 4

def d_hinge(v):
    return None

def d_hinge_loss_th(X, Y, th, th0):
    return None

def d_hinge_loss_th0(X, Y, th, th0):
    return None

def d_svm_obj_th(X, Y, th, th0, lam):
    return None

def d_svm_obj_th0(X, Y, th, th0, lam):
    return None

def svm_obj_grad(X, Y, th, th0, lam):
    return None


# Задание 5

def svm_grad_desc(data, labels, lam, eta, T):
    return None

