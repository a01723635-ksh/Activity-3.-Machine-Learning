import pandas as pd
import statsmodels.api as sm

# Financial dataset
data = {
    "Market_Return": [1.2, -0.8, 0.5, 2.1, -1.5, 0.9, -0.3, 1.7, -1.1, 0.4,
                      1.5, -0.6, 2.4, -1.8, 0.7, 1.1, -0.4, 1.9, -1.3, 0.2],

    "Stock_Return": [1.8, -1.1, 0.9, 2.7, -2.0, 1.4, -0.5, 2.2, -1.6, 0.8,
                     2.0, -0.7, 3.1, -2.4, 1.0, 1.6, -0.8, 2.5, -1.7, 0.5]
}

dataset = pd.DataFrame(data)

print(dataset)

# Define dependent and independent variables
X = dataset[["Market_Return"]]
Y = dataset["Stock_Return"]

# Add the intercept
X = sm.add_constant(X)

# Estimate the linear regression model
model = sm.OLS(Y, X).fit()

# Display the regression results
print(model.summary())