import numpy as np

trainingSamples = [[1, 2], [2, 4], [3, 5]]


def startW():
    return np.zeros(2)


def phi(x):
    return np.array([1, x])


def trainingLoss(w,x,y):
    return (np.dot(phi(x),w) - y)**2


def differential(w,x,y):
    return 2*(np.dot(phi(x),w) - y)*phi(x)


def gradDescent():
    w = startW()
    for _ in range(500):
        np.random.shuffle(trainingSamples)
        for x,y in trainingSamples:
            w = w - 0.1 * differential(w,x,y)
    return w


final_w = gradDescent()
print(f"Final weights: {final_w}")
