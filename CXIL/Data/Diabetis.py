from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pickle
import shap 

def train_and_save_interpretable_surrogate():
    #TODO Only returns one freature currently 
    #TODO Use Shap ? 
    # https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/linear_models/Explaining%20a%20model%20that%20uses%20standardized%20features.html

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Load the diabetes dataset
    diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

    # Use only one feature
    # TODO Why onlyuse one feature ? 
    #diabetes_X = diabetes_X[:, np.newaxis, 2]

    # Split the data into training/testing sets
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]

    # Split the targets into training/testing sets
    diabetes_y_train = diabetes_y[:-20]
    diabetes_y_test = diabetes_y[-20:]

    # Train the model using the training sets
    regr.fit(diabetes_X_train, diabetes_y_train)
    print("Coefficients: \n", regr.coef_)

    # Make predictions using the testing set
    diabetes_y_pred = regr.predict(diabetes_X_test)

    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))
    # The coefficient of determination: 1 is perfect prediction
    print("Coefficient of determination: %.2f" % r2_score(diabetes_y_test, diabetes_y_pred))
    pickle.dump(regr,open('./data/Interpretable_Surrogates/Diabetis/model_linear.pkl','wb'))
    summary= (mean_squared_error(diabetes_y_test, diabetes_y_pred),r2_score(diabetes_y_test, diabetes_y_pred))
    pickle.dump(summary,open('./data/Interpretable_Surrogates/Diabetis/summary.pkl','wb'))

    pickle.dump(regr.coef_,open('./data/Interpretable_Surrogates/Diabetis/coeff.pkl','wb'))

if __name__ =='__main__':
    print('Start')
    train_and_save_interpretable_surrogate()
    print('END')
