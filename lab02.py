import numpy as np


# Датасеты

def super_simple_separable_through_origin():
    X = np.array([[2, 3, 9, 12],
                  [5, 1, 6, 5]])
    y = np.array([[1, -1, 1, -1]])
    return X, y

def super_simple_separable():
    X = np.array([[2, 3, 9, 12],
                  [5, 2, 6, 5]])
    y = np.array([[1, -1, 1, -1]])
    return X, y

def xor():
    X = np.array([[1, 2, 1, 2],
                  [1, 2, 2, 1]])
    y = np.array([[1, 1, -1, -1]])
    return X, y

def xor_more():
    X = np.array([[1, 2, 1, 2, 2, 4, 1, 3],
                  [1, 2, 2, 1, 3, 1, 3, 3]])
    y = np.array([[1, 1, -1, -1, 1, 1, -1, -1]])
    return X, y

def dataset_1():
    X = np.array(
        [[ -2.97797707,  2.84547604,  3.60537239, -1.72914799, -2.51139524, 3.10363716, 2.13434789, 1.61328413, 2.10491257, -3.87099125, 3.69972003, -0.23572183, -4.19729119, -3.51229538, -1.75975746, -4.93242615, 2.16880073, -4.34923279, -0.76154262, 3.04879591, -4.70503877,  0.25768309,  2.87336016,  3.11875861, -1.58542576, -1.00326657, 3.62331703, -4.97864369, -3.31037331, -1.16371314 ],
        [ 0.99951218, -3.69531043, -4.65329654, 2.01907382, 0.31689211, 2.4843758, -3.47935105, -4.31857472, -0.11863976,  0.34441625, 0.77851176, 1.6403079, -0.57558913, -3.62293005, -2.9638734, -2.80071438, 2.82523704, 2.07860509, 0.23992709, 4.790368, -2.33037832, 2.28365246, -1.27955206, -0.16325247, 2.75740801, 4.48727808, 1.6663558, 2.34395397, 1.45874837, -4.80999977 ]])
    y = np.array([[-1., -1., -1., -1., -1., -1.,  1.,  1.,  1., -1., -1., -1., -1., -1.,  1., -1.,  1., -1., -1., -1.,  1.,  1.,  1.,  1.,  1., -1., -1., -1., -1., -1.]])
    return X, y

def dataset_2():
    X = np.array(
        [[ -2.97797707, 2.84547604, 3.60537239, -1.72914799, -2.51139524, 3.10363716, 2.13434789, 1.61328413, 2.10491257, -3.87099125, 3.69972003, -0.23572183, -4.19729119, -3.51229538, -1.75975746, -4.93242615, 2.16880073, -4.34923279, -0.76154262, 3.04879591, -4.70503877, 0.25768309, 2.87336016, 3.11875861, -1.58542576, -1.00326657, 3.62331703, -4.97864369, -3.31037331, -1.16371314],
        [ 0.99951218, -3.69531043, -4.65329654,  2.01907382,  0.31689211, 2.4843758, -3.47935105, -4.31857472, -0.11863976, 0.34441625, 0.77851176, 1.6403079, -0.57558913, -3.62293005, -2.9638734, -2.80071438, 2.82523704, 2.07860509, 0.23992709, 4.790368, -2.33037832, 2.28365246, -1.27955206, -0.16325247, 2.75740801, 4.48727808, 1.6663558, 2.34395397, 1.45874837, -4.80999977]])
    y = np.array([[ -1., -1., 1., 1., -1., -1., -1., 1., 1., 1., -1., 1., 1., -1., 1., 1., 1., -1., -1., -1., 1., -1., 1., -1., 1., -1., -1., 1., 1., 1.]])
    return X, y

def perceptron(data, labels, tau=100):
    d, n = data.shape
    th = np.zeros((d, 1))
    th0 = [[0]]

    for t in range(int(tau)):
        #print(t)
        for i in range(n):
            datai = data[:, [i]]
            #print("if ", labels[0][i] * (np.dot(th.T, datai) + th0)[0])
            if labels[0][i] * (np.dot(th.T, datai) + th0)[0] <= 0:
                th = th + np.dot(labels[0][i], datai)
                #print(th)
                th0 += labels[0][i]
                #print(th0)

    return th, th0
      
def averaged_perceptron(data, labels, tau=100):
    d, n = data.shape
    th = np.zeros((d, 1))
    th0 = [[0]]
    ths = np.zeros((d, 1))
    th0s = [[0]]

    for t in range(int(tau)):
        #print(t)
        for i in range(n):
            datai = data[:, [i]]
            #print("if ", labels[0][i] * (np.dot(th.T, datai) + th0)[0])
            if labels[0][i] * (np.dot(th.T, datai) + th0)[0] <= 0:
                th = th + np.dot(labels[0][i], datai)
                #print(th)
                th0 += labels[0][i]
                #print(th0)
            ths = ths + th
            #print(ths)
            th0s = th0s + th0
            #print(th0s)
    return ths/int(tau)/n, th0s/int(tau)/n

def eval_classifier(learner, data_train, labels_train, data_test, labels_test):
    th, th0 = learner(data_train, labels_train)
    #print(th, th0)
    dists = np.dot(th.T, data_test) + th0
    #print(dists)
    preds = np.where(dists > 0, 1, -1)
    #print(preds)

    return np.mean(preds == labels_test)

def data_gen(n):
    rng = np.random.default_rng()
    d = 2
    th = rng.random(size=(d, 1))
    #print(th)
    th0 = rng.integers(low=-100, high=100)
    #print(th0)
    data = rng.standard_normal((d, int(n)))
    #print(data)
    dists = np.dot(th.T, data) + th0
    labels = np.where(dists > 0, 1.0, -1.0)
    return data, labels

def eval_learning_alg(learner, data_gen, n_train, n_test, iter):
    scores = []
    for i in range(iter):
        #print(i)
        dataTrn, labelsTrn = data_gen(n_train)
        dataTst, labelsTst = data_gen(n_test)
        scores.append(eval_classifier(learner, dataTrn, labelsTrn, dataTst, labelsTst))

    return np.mean(scores)                                    


def xval_learning_alg(learner, data, labels, k):
    d, n = data.shape
    if (d % k == 0):
        for di in range (k-1):
            for j in range(k-1):
                d_minus_j = 
    return None

print(perceptron(super_simple_separable()[0], super_simple_separable()[1]))
print(perceptron(super_simple_separable_through_origin()[0], super_simple_separable_through_origin()[1]))

print(averaged_perceptron(super_simple_separable()[0], super_simple_separable()[1]))
print(averaged_perceptron(super_simple_separable_through_origin()[0], super_simple_separable_through_origin()[1]))

print(eval_classifier(perceptron, super_simple_separable()[0], super_simple_separable()[1], super_simple_separable_through_origin()[0], super_simple_separable_through_origin()[1]))

print(eval_learning_alg(perceptron, data_gen, 5, 15, 20))