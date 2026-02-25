import numpy as np

trainingSamples = [[1, 3, 1], [2, 5,  -1], [3, 1, 1]]
# x1, x2, y

def startW():
    return np.zeros(2)


def phi(x1,x2):
    return np.array([x1, x2])


def trainingLoss(w):
    return (
        1.0
        / len(trainingSamples)
        * sum(max(1 - np.dot(w, phi(x1,x2))*y,0)  for x1, x2, y in trainingSamples)
    )


def differential(w):
    return (
        1.0
        / len(trainingSamples)
        * sum( (0 if (1 - np.dot(w, phi(x1,x2))*y) <=0 else -1*phi(x1,x2)*y)  for x1, x2, y in trainingSamples)
    )


def gradDescent():
    w = startW()
    for _ in range(500):
        w = w - 0.1 * differential(w)
    return w


final_w = gradDescent()
print(f"Final weights: {final_w}")
