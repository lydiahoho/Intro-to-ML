# You are not allowed to import any additional packages/libraries.
import numpy as np
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

class LogisticRegression:
    def __init__(self, learning_rate=0.1, iteration=100):
        self.learning_rate = learning_rate
        self.iteration = iteration
        self.weights = None
        self.intercept = None

    # This function computes the gradient descent solution of logistic regression.
    def fit(self, X, y):
        row, col = X.shape
        gw, gi= np.zeros(col), 0
        # gw[:]=0.3
        XT = np.transpose(X)

        for _ in range(self.iteration):
            z = np.dot(X, gw)+ gi
            sig = self.sigmoid(z)
            w = np.dot(XT, (sig-y)) / row
            i = np.sum(sig-y) / row
            if(_>2100):
                 self.learning_rate = 0.004
    
            gw -= self.learning_rate* w *0.1
            gi -= self.learning_rate* i *2
                        
        self.weights, self.intercept = gw, gi
            
    # This function takes the input data X and predicts the class label y according to your solution.
    def predict(self, X):
        z = np.dot(X, self.weights)+ self.intercept
        sig = self.sigmoid(z)
        sol = np.where(sig >= 0.5, 1, 0)
        return sol

    # This function computes the value of the sigmoid function.
    def sigmoid(self, x):
        return 1 / (1 +  np.exp(-x))
        

class FLD:
    def __init__(self):
        self.w = None
        self.m0 = None
        self.m1 = None
        self.sw = None
        self.sb = None
        self.slope = None

    # This function computes the solution of Fisher's Linear Discriminant.
    def fit(self, X, y):
        
        self.m0 = np.mean(X[y==0], 0)
        self.m1 = np.mean(X[y==1], 0)
        b0 = X[y==0] - self.m0
        b1 = X[y==1] - self.m1
        self.sw = np.dot(np.transpose(b0), b0) + np.dot(np.transpose(b1), b1)
        self.sb = np.outer((self.m1 - self.m0), (self.m1 - self.m0))
        sw_inv = np.linalg.inv(self.sw)
        eigen, eigen_vec = np.linalg.eig(np.dot(sw_inv, self.sb)) 
        self.w = eigen_vec[:, np.argmax(eigen)]
        self.slope = self.w[1] /self.w[0]
 

    # This function takes the input data X and predicts the class label y by comparing the distance between the projected result of the testing data with the projected means (of the two classes) of the training data.
    # If it is closer to the projected mean of class 0, predict it as class 0, otherwise, predict it as class 1.
    def predict(self, X):
        py = np.dot(X, self.w)
        m0 = np.dot(self.m0, self.w)
        m1 = np.dot(self.m1, self.w)
        y_pred = np.where((np.abs(py - m0) < np.abs(py - m1)), 0, 1)
        return y_pred

    # This function plots the projection line of the testing data.
    # You don't need to call this function in your submission, but you have to provide the screenshot of the plot in the report.
    def plot_projection(self, X,y_pred):
        x_vals = np.linspace(0, 100, 100)
        mean = (self.m0 + self.m1) /2
        y_vals = self.slope * (x_vals-mean[0]) + mean[1]
        plt.figure(figsize=(7,7))
        plt.plot(x_vals, y_vals, color='k', linestyle='-', linewidth=1.5)
        colors = ['blue' if label == 0 else 'red' for label in y_pred]
        plt.scatter(X[:, 0], X[:, 1], c=colors, marker='o')  
        intercept = mean[1] - self.slope * mean[0]
        for x, y in zip(X[:, 0], X[:, 1]):
            proj_x = (x + self.slope * y - self.slope * intercept) / (1 + self.slope**2)
            proj_y = self.slope * proj_x + intercept
            plt.plot([x, proj_x], [y, proj_y], 'c-', lw=0.5)    
        # plt.scatter([self.m0[0], self.m1[0]], [self.m0[1], self.m1[1]], c='blue', marker='x', s=200, label='Class Means')
        plt.xlim(0, 150)  
        plt.ylim(50, 200)  
        plt.xlabel('Age')
        plt.ylabel('Thalach')
        plt.text(70, 190, f'Slope: {self.slope:.2f}, Intercept: {mean[0]:.2f}, {mean[1]:.2f}')

        plt.title(f'Projection Line of Testing Data')
        plt.legend([])

        plt.show()
        
     
# Do not modify the main function architecture.
# You can only modify the value of the arguments of your Logistic Regression class.
if __name__ == "__main__":
# Data Loading
    train_df = DataFrame(read_csv("train.csv"))
    test_df = DataFrame(read_csv("test.csv"))

# Part 1: Logistic Regression
    # Data Preparation
    # Using all the features for Logistic Regression
    X_train = train_df.drop(["target"], axis=1)
    y_train = train_df["target"]
    X_test = test_df.drop(["target"], axis=1)
    y_test = test_df["target"]
    
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    # Model Training and Testing
    LR = LogisticRegression(learning_rate=0.02, iteration=5000)
    LR.fit(X_train, y_train)
    y_pred = LR.predict(X_test)
    accuracy = accuracy_score(y_test , y_pred)
    print(f"Part 1: Logistic Regression")
    print(f"Weights: {LR.weights}, Intercept: {LR.intercept}")
    print(f"Accuracy: {accuracy}")
    # You must pass this assertion in order to get full score for this part.
    assert accuracy > 0.75, "Accuracy of Logistic Regression should be greater than 0.75"

# Part 2: Fisher's Linear Discriminant
    # Data Preparation
    # Only using two features for FLD
    X_train = train_df[["age", "thalach"]]
    y_train = train_df["target"]
    X_test = test_df[["age", "thalach"]]
    y_test = test_df["target"]
    
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    # Model Training and Testing
    FLD = FLD()
    FLD.fit(X_train, y_train)
    y_pred = FLD.predict(X_test)
    accuracy = accuracy_score(y_test , y_pred)
    print(f"Part 2: Fisher's Linear Discriminant")
    print(f"Class Mean 0: {FLD.m0}, Class Mean 1: {FLD.m1}")
    print(f"With-in class scatter matrix:\n{FLD.sw}")
    print(f"Between class scatter matrix:\n{FLD.sb}")
    print(f"w:\n{FLD.w}")
    print(f"Accuracy of FLD: {accuracy}")
    # You must pass this assertion in order to get full score for this part.
    assert accuracy > 0.65, "Accuracy of FLD should be greater than 0.65"
    FLD.plot_projection(X_test,y_pred)

