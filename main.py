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

    X = np.array([industrial_risk, management_risk, financial_flexibility, credibility,
                 competitiveness, operation_risk])

    return X
    # return rawData


# Define the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Define the loss function
def loss(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()


class LogisticRegression:
    def fit(self, x, y):
        pass

    def prediction(self, x):
        pass


if __name__ == "__main__":
    bankruptcyData("C:/Users/Eren/Downloads/Qualitative_Bankruptcy (250 "
                   "instances)/Qualitative_Bankruptcy/Qualitative_Bankruptcy.data.txt")

