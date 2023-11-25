import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt



def simple_linear_regression():
    # input array
    x = np.array([5,15,25,35,45,55])
    # needs to be two dimensional
    # reshape to 6 rows by 1 column (-1 auto-calculates the 6)
    x = x.reshape(-1, 1)

    print(f"Input array:\n {x}")
    print(f"Shape:\n {x.shape}")
    print()

    # output array
    y = np.array([5, 20, 14, 32, 22, 38])
    print(f"Output array:\n {y}")
    print(f"Shape:\n {y.shape}")
    print()

    # create a scatterplot for input data
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot()
    ax.scatter(x, y, color='g')


    # create model
    model = LinearRegression()
    print(f"Model: {model}")
    print()

    # compute coeficients
    model.fit(x, y)

    # b0 / intercept coedicient
    print(f"Intercept:\n {model.intercept_}")
    print()

    # b1 coeficients (in this case only one)
    print(f"Coeficients:\n {model.coef_}")
    print()

    # r squared value (how well model fits our data)
    # value closest to 1 means model fits data very well
    r_sq = model.score(x, y)
    print(f"R_sq:\n {r_sq}")
    print()

    # predict y values
    y_pred = model.predict(x)
    print(f"Predicted y values:\n {y_pred}")
    print("or:")
    # the same can be achieved by the following calculation
    # f(x) = b0 + b1 * x
    print(model.intercept_ + model.coef_ * x)
    print()

    # create a regression line
    plt.plot(x, y_pred, color='b')

    # create new set of input values and predict their output values based on our model
    x_new = np.arange(5).reshape(-1,1)
    print(f"New input array:\n {x_new}")
    print()

    y_new = model.predict(x_new)
    print(f"Predicted values:\n {y_new}")

    # show plot
    plt.rcParams.update({'font.size': 18})
    ax.set_xlabel("feature 1", fontsize=18)
    ax.set_ylabel("label", fontsize=18)
    ax.legend(["Data points", "Regression Line"])
    plt.show()

def multiple_linear_regression():
    # input array
    x = [[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]]
    x = np.array(x)
    print(f"Input array:\n {x}")
    print(f"Shape:\n {x.shape}")
    print()

    # output array
    y = [4, 5, 20, 14, 32, 22, 38, 43]
    y = np.array(y)
    print(f"Output array:\n {y}")
    print(f"Shape:\n {y.shape}")
    print()

    # create a scatterplot for input data
    feature_1 = [z[0] for z in x]
    feature_2 = [z[1] for z in x]

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(feature_1, feature_2, y, color='g')



    # create model
    model = LinearRegression()
    print(f"Model: {model}")
    print()

    # compute coeficients
    model.fit(x, y)

    # b0 / intercept coedicient
    print(f"Intercept:\n {model.intercept_}")
    print()

    # b1, b2.. coeficients (in this case two)
    print(f"Coeficients:\n {model.coef_}")
    print()

    # r squared value (how well model fits our data)
    # value closest to 1 means model fits data very well
    r_sq = model.score(x, y)
    print(f"R_sq:\n {r_sq}")
    print()

    # compute estimates (y values)
    y_est = model.predict(x)
    print(f"Predicted y estimates:\n {y_est}")
    print()

    # show estimates vs observed values
    print("Estimate".ljust(12), "Observed value")
    for i in range(len(y)):
        print(f"{y_est[i]:<12.8f}", f"{y[i]}")
    print()

    # create a regression line
    ax.plot(feature_1, feature_2, y_est, color='b')

    # create new set of input values and predict their output values based on our model
    x_new = np.arange(10).reshape(-1, 2)
    print(f"New input array:\n {x_new}")
    print()

    y_new = model.predict(x_new)
    print(f"Predicted values:\n {y_new}")

    # show plot
    plt.rcParams.update({'font.size': 18})
    ax.set_xlabel("feature 1", fontsize=18)
    ax.set_ylabel("feature 2", fontsize=18)
    ax.set_zlabel("label", fontsize=18)
    ax.legend(["Data points", "Regression Line"])

    plt.show()


def simple_polynomial_regression():
    # input array
    x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
    print(f"Input array:\n {x}")
    print(f"Shape:\n {x.shape}")
    print()

    # output array
    y = np.array([15, 11, 2, 8, 25, 32])
    print(f"Output array:\n {y}")
    print(f"Shape:\n {y.shape}")
    print()

    # create a scatterplot for input data
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot()
    ax.scatter(x, y, color='g')

    # create a PolynomialFeatures instance to serve as a data transformer (in this case for a quadratic model)
    transformer = PolynomialFeatures(degree=2, include_bias=False)

    x_ = transformer.fit_transform(x)
    print(f"Transformed input data:\n {x_}")
    print()

    # create model
    model = LinearRegression()
    print(f"Model: {model}")
    print()

    # compute coeficients with transfromed data
    model.fit(x_, y)

    # b0 / intercept coedicient
    print(f"Intercept:\n {model.intercept_}")
    print()

    # b1, b2.. coeficients (here, two for cubic representation of a single feature)
    print(f"Coeficients:\n {model.coef_}")
    print()

    # r squared value (how well model fits our data)
    # value closest to 1 means model fits data very well
    r_sq = model.score(x_, y)
    print(f"R_sq:\n {r_sq}")
    print()

    # compute estimates (y values)
    y_est = model.predict(x_)
    print(f"Predicted y estimates:\n {y_est}")
    print()

    # show estimates vs observed values
    print("Estimate".ljust(12), "Observed value")
    for i in range(len(y)):
        print(f"{y_est[i]:<12.8f}", f"{y[i]}")
    print()

    # create a regression line
    plt.plot(x, y_est, color='b')

    # create new set of input values and predict their output values based on our model
    x_new = np.arange(10).reshape(-1, 1)
    print(f"New input array:\n {x_new}")
    print()

    x_new_ = transformer.fit_transform(x_new)
    print(f"Transformed input data:\n {x_new_}")
    print()

    y_new = model.predict(x_new_)
    print(f"Predicted values:\n {y_new}")

    # show plot
    plt.rcParams.update({'font.size': 18})
    ax.set_xlabel("feature 1", fontsize=18)
    ax.set_ylabel("label", fontsize=18)
    ax.legend(["Data points", "Regression Line"])
    plt.show()


def multiple_polynomial_regression():
    # input array
    x = [[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]]
    x = np.array(x)
    print(f"Input array:\n {x}")
    print(f"Shape:\n {x.shape}")
    print()

    # output array
    y = [4, 5, 20, 14, 32, 22, 38, 43]
    y = np.array(y)
    print(f"Output array:\n {y}")
    print(f"Shape:\n {y.shape}")
    print()

    # create a scatterplot for input data
    feature_1 = [z[0] for z in x]
    feature_2 = [z[1] for z in x]

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(feature_1, feature_2, y, color='g')

    # create a PolynomialFeatures instance to serve as a data transformer (in this case for a quadratic model)
    transformer = PolynomialFeatures(degree=2, include_bias=False)

    x_ = transformer.fit_transform(x)
    print(f"Transformed input data:\n {x_}")
    print()

    # create model
    model = LinearRegression()
    print(f"Model: {model}")
    print()

    # compute coeficients with transfromed data
    model.fit(x_, y)

    # b0 / intercept coedicient
    print(f"Intercept:\n {model.intercept_}")
    print()

    # b1, b2.. coeficients (here, 5 for cubic representation of a 2 features; the fifth value is a mix between features)
    print(f"Coeficients:\n {model.coef_}")
    print()

    # r squared value (how well model fits our data)
    # value closest to 1 means model fits data very well
    r_sq = model.score(x_, y)
    print(f"R_sq:\n {r_sq}")
    print()

    # compute estimates (y values)
    y_est = model.predict(x_)
    print(f"Predicted y estimates:\n {y_est}")
    print()

    # show estimates vs observed values
    print("Estimate".ljust(12), "Observed value")
    for i in range(len(y)):
        print(f"{y_est[i]:<12.8f}", f"{y[i]}")
    print()

    # create a regression line
    ax.plot(feature_1, feature_2, y_est, color='b')

    # create new set of input values and predict their output values based on our model
    x_new = np.arange(10).reshape(-1, 2)
    print(f"New input array:\n {x_new}")
    print()

    x_new_ = transformer.fit_transform(x_new)
    print(f"Transformed input data:\n {x_new_}")
    print()

    y_new = model.predict(x_new_)
    print(f"Predicted values:\n {y_new}")

    # show plot
    plt.rcParams.update({'font.size': 18})
    ax.set_xlabel("feature 1", fontsize=18)
    ax.set_ylabel("feature 2", fontsize=18)
    ax.set_zlabel("label", fontsize=18)
    ax.legend(["Data points", "Regression Line"])
    plt.show()


if __name__ == '__main__':
    while True:
        print("""
*********************************
1. Simple Linear Regression
2. Multiple Linear Regression
3. Simple Polynomial Regression
4. Multiple Polynomial Regression
0. Exit
*********************************""")
        option = input("> ")

        match option:
            case "1":
                simple_linear_regression()
            case "2":
                multiple_linear_regression()
            case "3":
                simple_polynomial_regression()
            case "4":
                multiple_polynomial_regression()
            case "0":
                exit()
            case _:
                print("Invalid option.")
