import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Return fitted model parameters to the dataset at datapath for each choice in degrees.
#Input: datapath as a string specifying a .txt file, degrees as a list of positive integers.
#Output: paramFits, a list with the same length as degrees, where paramFits[i] is the list of
#coefficients when fitting a polynomial of d = degrees[i].
def main(datapath, degrees):
    paramFits = []

    #fill in
    #read the input file, assuming it has two columns, where each row is of the form [x y] as
    x,y = np.loadtxt(datapath, delimiter='\t',dtype = float, unpack=True)

    #in poly.txt.
    #iterate through each n in degrees, calling the feature_matrix and least_squares functions to solve
    #Beta = (X.TX)inv X.Ty

    for i in degrees:
        X = feature_matrix(x,i)
        paramFits.append(least_squares(X,y))
    #for the model parameters in each case. Append the result to paramFits each time.
    return paramFits
def gety(datapath):
    x,y = np.loadtxt(datapath, delimiter='\t',dtype = float, unpack=True)

    return y, x


#Return the feature matrix for fitting a polynomial of degree d based on the explanatory variable
#samples in x.
#Input: x as a list of the independent variable samples, and d as an integer.
#Output: X, a list of features for each sample, where X[i][j] corresponds to the jth coefficient
#for the ith sample. Viewed as a matrix, X should have dimension #samples by d+1.
def feature_matrix(x, d):
    X = []
    #newArr = np.array(newArr)
    #fill in
    #There are several ways to write this function. The most efficient would be a nested list comprehension
    #which for each sample in x calculates x^d, x^(d-1), ..., x^0.
    for j in x:
        newArr = []
        for i in range(d, -1, -1):
            newArr.append((j ** i))
        X.append(newArr)
    #X = np.array(newArr)
    #X = X.reshape((len(x), d+1))
    return X


#Return the least squares solution based on the feature matrix X and corresponding target variable samples in y.
#Input: X as a list of features for each sample, and y as a list of target variable samples.
#Output: B, a list of the fitted model parameters based on the least squares solution.
def least_squares(X, y):
    X = np.array(X)
    y = np.array(y)
    #A = np.matmul(((np.linalg.inv(np.matmul((X.T), X))), (X.T)), y)

    # Beta = (X.TX)inv X.Ty
    #fill in
    #Use the matrix algebra functions in numpy to solve the least squares equations. This can be done in just one line.
    #X.T

    A = X.T
    Be = A@X
    C = np.linalg.inv(Be)
    D = C@A
    B = D@y
   # print(B)
    B = B.tolist()

    return B




if __name__ == '__main__':
    datapath = 'poly.txt'
    degrees = [1,2,3,4,5]

    paramFits = main(datapath, degrees)
    y,x = gety(datapath)
    print(paramFits)

    for list1 in paramFits:
        newlist = []
        for i in list1:
            newlist.append(np.format_float_positional(i, precision=6, unique=False, fractional=False, trim='k'))

    # Create the plot
    plt.scatter(x, y, label='data')
    x.sort()
    for list1 in paramFits:
        newy = feature_matrix(x,len(list1)-1)
        newy = np.array(newy)
        y = np.dot(newy,list1)
        label = 'd = ' + str(len(list1) - 1)
        if (len(list1)-1 == 3):
            yint = (np.interp(2,x,y))
            plt.plot(2, yint, 'r*', label='New Data Point X = 2')

        plt.plot(x, y, label= label)
    # Add a title
    plt.title('Fitted model for varying degrees')

    # Add X and y Label
    plt.xlabel('x axis')
    plt.ylabel('y axis')

    # Add a grid
    plt.grid(alpha=.4, linestyle='--')

    # Add a Legend
    plt.legend()

    # Show the plot
    plt.show()

