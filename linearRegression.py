import numpy as np

trainingSamples = [
    [1, 2],
    [2, 4],
    [3, 5]
]

def startW():
    return np.zeros(2)

def phi(x):
    return np.array([1, x])

def trainingLoss(w):
    return 1.0 / len(trainingSamples) * sum((np.dot(phi(x), w) - y)**2 for x, y in trainingSamples)

def differential(w):
    return 1.0 / len(trainingSamples) * sum(2 * (np.dot(phi(x), w) - y) * phi(x) for x, y in trainingSamples)

def gradDescent():
    w = startW()  
    for _ in range(500):
        w = w - 0.1 * differential(w)
    return w

final_w = gradDescent()
print(f"Final weights: {final_w}")