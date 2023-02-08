import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


def bankruptcyData(directory):
    rawData = np.loadtxt(directory, dtype=str, delimiter=',')

    input_mapping = {'P': 2, 'A': 1, 'N': 0}
    output_mapping = {'B': 0, 'NB': 1}

    industrial_risk = np.array([input_mapping[item] for item in rawData[:, 0]])
    management_risk = np.array([input_mapping[item] for item in rawData[:, 1]])
    financial_flexibility = np.array([input_mapping[item] for item in rawData[:, 2]])
    credibility = np.array([input_mapping[item] for item in rawData[:, 3]])
    competitiveness = np.array([input_mapping[item] for item in rawData[:, 4]])
    operation_risk = np.array([input_mapping[item] for item in rawData[:, 5]])
    outcome = np.array([output_mapping[item] for item in rawData[:, 6]])

    X = np.array([industrial_risk, management_risk, financial_flexibility, credibility,
                  competitiveness, operation_risk])

    return outcome, X.T
    # return rawData


# Define the sigmoid function
def activation(x):
    return 1 / (1 + np.exp(-x))


# Define the loss function
def loss(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()


# y is the vector with the actual values of the function (bankrupt-not bankrupt)
# x is the data
# w is the vector of weights
def cost(w, x, y):
    z = np.dot(x, w)
    return np.mean(y * np.log1p(np.exp(-z)) + (1 - y) * np.log1p(np.exp(z)))


# y is the vector with the actual values of the function (bankrupt-not bankrupt)
# x is the data
# w is the vector of weights
def gradient(w, x, y):
    N = x.shape[0]
    D = x.shape[1]
    return np.dot(x.T, activation(np.dot(x, w)) - y) / N


def GradientDescent(x, y, lr=.01, eps=1e-2, w_init=None):
    N = x.shape[0]
    D = x.shape[1]
    if w_init is not None:
        w = w_init
    else:
        w = np.zeros(D)

    g = np.inf
    iterations = 0
    gra = list()
    w1 = list()
    iters = list()

    while abs(np.linalg.norm(g)) > eps:#and iterations < 10000:
        g = gradient(w, x, y)
        w -= lr*g
        w1.append(w[0])
        iterations += 1
        iters.append(iterations)
        gra.append(g)

    plt.plot(np.array(iters), np.array(w1))
    plt.plot(np.array(iters), gra)
    plt.show()
    return w


class LogisticRegressionCls:
    def fit(self, x, y):
        pass

    def prediction(self, x):
        pass


if __name__ == "__main__":
    outcome, X = bankruptcyData("C:/Users/Eren/Downloads/Qualitative_Bankruptcy (250 "
                                "instances)/Qualitative_Bankruptcy/Qualitative_Bankruptcy.data.txt")
    print(X)
    print(X.shape)

    print(outcome)
    print(outcome.T)

    X_train, X_test, y_train, y_test = train_test_split(X, outcome, test_size=0.2, random_state=42)

    # Maybe they all play an equal role?
    w = np.array([1, 1, 1, 1, 1, 1])
    print(cost(w, X, outcome))

    w = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    print("sk.learn")
    print("_______________________________________")
    weights = clf.coef_
    print(weights)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    print("\nCustom class")
    print("_______________________________________")
    w2 = GradientDescent(X_train, y_train)
    print(w2)
    print(cost(w2, X, outcome))
    print(cost(weights.T, X, outcome))
