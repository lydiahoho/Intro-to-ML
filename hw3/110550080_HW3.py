# You are not allowed to import any additional packages/libraries.
import numpy as np
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# This function computes the gini impurity of a label array.
def gini(y):
    total = len(y)
    p0 = np.count_nonzero(y == 0) / total
    p1 = np.count_nonzero(y == 1) / total
    if p0 == 0 or p1 ==0:
        return 0
    else:
        ig = 1 - (p0*p0 + p1*p1)
        return ig

# This function computes the entropy of a label array.
def entropy(y):
    total = len(y)
    p0 = np.count_nonzero(y == 0) / total
    p1 = np.count_nonzero(y == 1) /total
    if p0 == 0 or p1 ==0:
        return 0
    else:
        ih = -(p0 * np.log2(p0) + p1 * np.log2(p1))
        return ih

class Node:
     def __init__(self):
        self.classes = 0
        self.feature = 0
        self.threshold = 0
        self.left = None
        self.right = None
                 
# The decision tree classifier class.
# Tips: You may need another node class and build the decision tree recursively.
class DecisionTree():
    def __init__(self, criterion='gini', max_depth=None):
        self.criterion = criterion
        self.max_depth = max_depth 
        self.root = None
        self.feat = []
        
    # This function computes the impurity based on the criterion.
    def impurity(self, y):
        if self.criterion == 'gini':
            return gini(y)
        elif self.criterion == 'entropy':
            return entropy(y)
    
    # This function fits the given data using the decision tree algorithm.
    def fit(self, X, y):
        self.root = self.create(X, y, 0)
        # print(self.root.threshold)
        
    def create(self, x, y, depth):
        # predict the class of this node
        node = Node()
        num = np.bincount(y)
        predicted_class = np.argmax(num)
        # print(num)
        node.classes = predicted_class
        
        if depth == self.max_depth or len(np.unique(y)) == 1:
            return node
        
        # find best threshold
        feature = x.shape[1]
        best_imp = np.inf
        threshold = None
        feat = None
        for i in range(feature):  
            data = np.unique(x[:, i])
            for v in data:
                idx_s = x[:, i] <= v
                idx_b = ~idx_s
                y_small = y[idx_s]
                y_big = y[idx_b]
                # print(y_big)
                if len(y_small) == 0 or len(y_big) == 0:
                    # print(len(y_small))
                    continue
                
                imp = (len(y_small) * self.impurity(y_small) + len(y_big) * self.impurity(y_big)) / len(y)
                if imp < best_imp:
                    best_imp = imp
                    threshold = v
                    feat = i
                    
        if feat is None:
            return node 
                    
        # splitt left and right 
        left = x[:, feat] <= threshold
        right = ~left
        node.feature = feat
        self.feat.append(feat)
        node.threshold = threshold
        # print(threshold)
        node.left = self.create(x[left], y[left], depth+1)
        node.right = self.create(x[right], y[right], depth+1)
        return node
    
    def pre(self, x, node):
        if node.left is None and node.right is None:
            return node.classes
        if x[node.feature] <= node.threshold:
            return self.pre(x, node.left) 
        else:
            return self.pre(x, node.right)
                   
                               
    # This function takes the input data X and predicts the class label y according to your trained model.
    def predict(self, X):
        y_pred = []
        for x in X:
            yp = self.pre(x, self.root)
            y_pred.append(yp) 
        return y_pred    
    
    # This function plots the feature importance of the decision tree.
    def plot_feature_importance_img(self, columns):
        categories = ['age', 'sex', 'cp', 'fbs', 'thalach', 'thal']
        values = np.bincount(self.feat)
        # print(values)
        plt.barh(categories, values)
        plt.title('Feature Importance')
        plt.show()

# The AdaBoost classifier class.
class AdaBoost():
    def __init__(self, criterion='gini', n_estimators=200):
        self.criterion = criterion 
        self.n_estimators = n_estimators
        self.classifier = []
        self.alpha = []

    # This function fits the given data using the AdaBoost algorithm.
    # You need to create a decision tree classifier with max_depth = 1 in each iteration.
    def fit(self, X, y):
        x_train = X
        y_train = y
        num = 200
        d = np.ones(len(y)) / len(y)
        for _ in range(self.n_estimators):
            cl = DecisionTree(criterion= self.criterion, max_depth=1)
            cl.fit(x_train, y_train)
            pre = cl.predict(X)
            miss = d[pre != y]
            err = np.sum(miss)
            # print(err)
            if err>0.5:
                err = 1-err
                pre = np.subtract(1,pre)
                # tmp = cl.root.left 
                # cl.root.left = cl.root.right
                # cl.root.right = tmp
                cl.root.left.classes = 1- cl.root.left.classes
                cl.root.right.classes = 1- cl.root.right.classes
                # pre = cl.predict(X)
                # miss = d[pre != y]
                # e = err
                # err = np.sum(miss)
                # print(e,err)
                    
            if err == 0:
                err = 1e-10
                
            a = 0.5 * np.log((1-err) / err)
                  
            for i in range(len(d)):
                if y[i] == pre[i]:
                    d[i] *= np.exp(-a)
                else:
                    d[i] *= np.exp(a)        
            d /= sum(d)
        
            self.classifier.append(cl)
            self.alpha.append(a)
            selected_indices = np.random.choice(np.arange(len(y)), size=num, replace=False, p=d)
            x_train = X[selected_indices]
            y_train = y[selected_indices]
              

    # This function takes the input data X and predicts the class label y according to your trained model.
    def predict(self, X):
        prediction = np.zeros(len(X))
        for a, c in zip(self.alpha, self.classifier):
            yp = np.array(c.predict(X))
            pred= [1 if x == 1 else -1 for x in yp]
            prediction += np.multiply(a, pred)
        
        y_pred = prediction>0
        return y_pred
    
        

# Do not modify the main function architecture.
# You can only modify the value of the random seed and the the arguments of your Adaboost class.
if __name__ == "__main__":
# Data Loading
    train_df = DataFrame(read_csv("train.csv"))
    test_df = DataFrame(read_csv("test.csv"))
    X_train = train_df.drop(["target"], axis=1)
    y_train = train_df["target"]
    X_test = test_df.drop(["target"], axis=1)
    y_test = test_df["target"]

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

# Set random seed to make sure you get the same result every time.
# You can change the random seed if you want to.
    np.random.seed(0)

# Decision Tree
    print("Part 1: Decision Tree")
    data = np.array([0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1])
    print(f"gini of [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1]: {gini(data)}")
    print(f"entropy of [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1]: {entropy(data)}")
    tree = DecisionTree(criterion='gini', max_depth=7)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    print("Accuracy (gini with max_depth=7):", accuracy_score(y_test, y_pred))
    tree = DecisionTree(criterion='entropy', max_depth=7)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    print("Accuracy (entropy with max_depth=7):", accuracy_score(y_test, y_pred))
    # tree2 = DecisionTree(criterion='gini', max_depth=15)
    # tree2.fit(X_train, y_train)
    # tree2.plot_feature_importance_img(7)
    

# AdaBoost
    print("Part 2: AdaBoost")
    # Tune the arguments of AdaBoost to achieve higher accuracy than your Decision Tree.
    ada = AdaBoost(criterion='gini', n_estimators=200)
    ada.fit(X_train, y_train)
    y_pred = ada.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))


    
