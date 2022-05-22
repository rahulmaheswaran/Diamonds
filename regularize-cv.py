import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def main():
    #Importing dataset
    diamonds = pd.read_csv('diamonds.csv')

    #Feature and target matrices
    X = diamonds[['carat', 'depth', 'table', 'x', 'y', 'z', 'clarity', 'cut', 'color']]
    y = diamonds[['price']]

    #Training and testing split, with 25% of the data reserved as the test set
    X = X.to_numpy()
    y = y.to_numpy()
    [X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size=0.25, random_state=101)

    #Normalizing training and testing data
    [X_train, trn_mean, trn_std] = normalize_train(X_train)
    X_test = normalize_test(X_test, trn_mean, trn_std)

    #Define the range of lambda to test
    lmbda = np.logspace(-1.00,2,num=101)
    #lmbda = [1,100]

    MODEL = []
    MSE = []
    for l in lmbda:
        #Train the regression model using a regularization parameter of l
        model = train_model(X_train,y_train,l)

        #Evaluate the MSE on the test set
        mse = error(X_test,y_test,model)

        #Store the model and mse in lists for further processing
        MODEL.append(model)
        MSE.append(mse)

    ind = MSE.index(min(MSE))
    [lmda_best,MSE_best,model_best] = [lmbda[ind],MSE[ind],MODEL[ind]]

    print('Best lambda tested is ' + str(lmda_best) + ', which yields an MSE of ' + str(MSE_best))
    # Plot the MSE as a function of lmbda
    plt  # fill in
    plt.plot(lmbda, MSE, color='black', label='MSE vs Lmbda')
    plt.plot(lmda_best, MSE_best, 'r*', label='Best lambda')
    plt.grid(alpha=.4, linestyle='--')
    plt.xlabel('Lambda λ')
    plt.ylabel('Mean Squared Error')
    plt.title('MSE Model for best fitting Lambda')
    plt.legend()
    plt.show()

#Code to Predict Price using model_best
    x_new = np.array([[0.25,60, 55,4,3,2,5,3,3]])
    X_new = normalize_test(x_new,trn_mean, trn_std)
    price = model_best.predict(X_new)
    print(np.format_float_positional(price, precision=6, unique=False, fractional=False, trim='k'))




    return model_best


#Function that normalizes features in training set to zero mean and unit variance.
#Input: training data X_train
#Output: the normalized version of the feature matrix: X, the mean of each column in
#training set: trn_mean, the std dev of each column in training set: trn_std.
def normalize_train(X_train):

    #fill in
    num_col = len(X_train[0])
    mean = []
    std = []
    X = X_train
    for i in range(num_col):
        mean.append(np.mean(X_train[:,i], axis=0))
        std.append(np.std(X_train[:,i], axis=0))
        X[:,i] = ((X_train[:,i] - mean[i])/std[i])


    return X, mean, std


#Function that normalizes testing set according to mean and std of training set
#Input: testing data: X_test, mean of each column in training set: trn_mean, standard deviation of each
#column in training set: trn_std
#Output: X, the normalized version of the feature matrix, X_test.
def normalize_test(X_test, trn_mean, trn_std):

    num_col = len(X_test[0])
    X = X_test

    for i in range(num_col):
        X[:,i] = ((X_test[:,i] - trn_mean[i])/trn_std[i])


    return X



#Function that trains a ridge regression model on the input dataset with lambda=l.
#Input: Feature matrix X, target variable vector y, regularization parameter l.
#Output: model, a numpy object containing the trained model.
def train_model(X,y,l):

    #fill in
    model = Ridge(alpha=l, fit_intercept=True)
    model.fit(X,y)


    return model


#Function that calculates the mean squared error of the model on the input dataset.
#Input: Feature matrix X, target variable vector y, numpy model object
#Output: mse, the mean squared error
def error(X,y,model):
    #Fill in
    ypred = model.predict(X)
    mse = mean_squared_error(y, ypred)

    return mse

if __name__ == '__main__':
    model_best = main()
    #We use the following functions to obtain the model parameters instead of model_best.get_params()
    #print(model_best.coef_)

    #Uncomment for 6 sig figs
    '''
        for list1 in model_best.coef_:
        newlist = []
        for i in list1:
            newlist.append(np.format_float_positional(i, precision=6, unique=False, fractional=False, trim='k'))
        print(newlist)
    
    '''
    print(model_best.coef_)

    print(np.format_float_positional(model_best.intercept_, precision=6, unique=False, fractional=False, trim='k'))



