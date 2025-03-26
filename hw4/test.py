from sklearn.model_selection import GridSearchCV

# Define the parameter grid for each kernel
param_grid_linear = {'C': [0.1, 1, 10, 100]}
param_grid_poly = {'C': [0.1, 1, 10], 'degree': [2, 3, 4]}
param_grid_rbf = {'C': [0.1, 1, 10], 'gamma': [0.1, 1, 10]}

# Perform grid search for each kernel
grid_linear = GridSearchCV(SVC(kernel='precomputed'), param_grid_linear)
grid_linear.fit(gram_matrix(X_train, X_train, linear_kernel), y_train)
best_params_linear = grid_linear.best_params_

# Repeat the same for polynomial and RBF kernels
# ...

# Use the best parameters to train the final models and evaluate
# ...
