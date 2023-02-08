import numpy as np
from matplotlib import pyplot as plt


def bankruptcyData(directory):
    rawData = np.loadtxt(directory, dtype=str, delimiter=',')

    input_mapping = {'P': 2, 'A': 1, 'N': 0}
    output_mapping = {'B': 1, 'NB': 0}

    industrial_risk = np.array([input_mapping[item] for item in rawData[:, 0]])
    management_risk = np.array([input_mapping[item] for item in rawData[:, 1]])
    financial_flexibility = np.array([input_mapping[item] for item in rawData[:, 2]])
    credibility = np.array([input_mapping[item] for item in rawData[:, 3]])
    competitiveness = np.array([input_mapping[item] for item in rawData[:, 4]])
    operation_risk = np.array([input_mapping[item] for item in rawData[:, 5]])

    outcome = np.array([output_mapping[item] for item in rawData[:, 6]])

    x = np.linspace(-1, 1, 50)
    y = np.linspace(0, 1, 50)
    print(len(industrial_risk))
    print(len(outcome))
    plt.scatter(management_risk[1], industrial_risk[1])
    plt.show()
    print(outcome)

    X = np.array([[industrial_risk], [management_risk], [financial_flexibility], [credibility],
                 [competitiveness], [operation_risk]])

    return X
    # return rawData


# Define the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Define the loss function
def loss(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()


def gradient(self, x, y):
    N, D = x.shape
    yh = sigmoid(np.dot(x, self.w))    # predictions  size N
    grad = np.dot(x.T, yh - y)/N        # divide by N because cost is mean over N points
    return grad


class LogisticRegression:

    def __init__(self, add_bias=True, learning_rate=.1, epsilon=1e-4, max_iters=1e5, verbose=False):
        self.add_bias = add_bias
        self.learning_rate = learning_rate
        self.epsilon = epsilon  # to get the tolerance for the norm of gradients
        self.max_iters = max_iters  # maximum number of iteration of gradient descent
        self.verbose = verbose

    def fit(self, x, y):
        if x.ndim == 1:
            x = x[:, None]
        if self.add_bias:
            N = x.shape[0]
            x = np.column_stack([x, np.ones(N)])
        N, D = x.shape
        self.w = np.zeros(D)
        g = np.inf
        t = 0
        # the code snippet below is for gradient descent
        while np.linalg.norm(g) > self.epsilon and t < self.max_iters:
            g = self.gradient(x, y)
            self.w = self.w - self.learning_rate * g
            t += 1

        if self.verbose:
            print(f'terminated after {t} iterations, with norm of the gradient equal to {np.linalg.norm(g)}')
            print(f'the weight found: {self.w}')
        return self

    def predict(self, x):
        if x.ndim == 1:
            x = x[:, None]
        Nt = x.shape[0]
        if self.add_bias:
            x = np.column_stack([x, np.ones(Nt)])
        yh = sigmoid(np.dot(x, self.w))  # predict output
        return yh


LogisticRegression.gradient = gradient  # initialize the gradient method of the LogisticRegression class with gradient function


if __name__ == "__main__":
    X = bankruptcyData("C:/Users/Eren/Downloads/Qualitative_Bankruptcy (250 "
                       "instances)/Qualitative_Bankruptcy/Qualitative_Bankruptcy.data.txt")

    reg = LogisticRegression()
    reg.fit(X[:1], y=np.array([0, 1]))

