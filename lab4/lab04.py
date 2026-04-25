import numpy as np

# Далее, эти переменные определены ровно как на лекции:
# X - матрица d x n
# Y - матрица 1 x n
# th - матрица d x 1
# th0 - матрица 1 x 1


# Задание 2a

def lin_reg(X, th, th0):
    return None

def square_loss(X, Y, th, th0):
    return None

def mean_square_loss(X, Y, th, th0):
    return None


# Задание 2b

def d_lin_reg_th(X, th, th0):
    return None

def d_square_loss_th(X, Y, th, th0):
    return None

def d_mean_square_loss_th(X, Y, th, th0):
    return None


# Задание 2c

def d_lin_reg_th0(X, th, th0):
    return None

def d_square_loss_th0(X, Y, th, th0):
    return None

def d_mean_square_loss_th0(X, Y, th, th0):
    return None


# Задание 2d

def ridge_obj(X, Y, th, th0, lam):
    return None

def d_ridge_obj_th(X, Y, th, th0, lam):
    return None

def d_ridge_obj_th0(X, Y, th, th0, lam):
    return None




# Задание 3

def stoc_grad_desc(X, Y, J, dJ, w0, eta, T):
    return None


# Датасет и соответствующие ей функции J и dJ для Задания 3

def downwards_line():
    X = np.array([[0.0, 0.1, 0.2, 0.3, 0.42, 0.52, 0.72, 0.78, 0.84, 1.0],
                  [1.0, 1.0, 1.0, 1.0, 1.0,  1.0,  1.0,  1.0,  1.0,  1.0]])
    Y = np.array([[0.4, 0.6, 1.2, 0.1, 0.22, -0.6, -1.5, -0.5, -0.5, 0.0]])
    return X, Y

X, Y = downwards_line()

def J(Xi, Yi, w):
    # перевод из формата (1-augmented X, Y, th) в (separated X, Y, th, th0) формат
    return float(ridge_obj(Xi[:-1,:], Yi, w[:-1,:], w[-1:,:], 0))

def dJ(Xi, Yi, w):
    grad_th = d_ridge_obj_th(Xi[:-1,:], Yi, w[:-1,:], w[-1:,:], 0)
    grad_th0 = d_ridge_obj_th0(Xi[:-1,:], Yi, w[:-1,:], w[-1:,:], 0)
    return np.vstack([grad_th, grad_th0])

