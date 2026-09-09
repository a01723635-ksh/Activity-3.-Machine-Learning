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
# Interpretation of Results:
# The estimated intercept is approximately 0.0289.
# The coefficient for Market Return is approximately 1.3414.
# This means that a 1 percentage-point increase in Market Return is associated
# with approximately a 1.34 percentage-point increase in Stock Return.
#
# The R-squared is approximately 0.993, which means that about 99.3% of the
# variation in Stock Return is explained by Market Return in this simulated dataset.
#
# The p-value for Market Return is below 0.05, which indicates that Market Return
# is statistically significant. The intercept has a p-value of approximately
# 0.412, so the intercept is not statistically significant.

# Comparison with Excel:
# Excel and Python produced practically the same results.
# Excel: Intercept = 0.0289, Market Return coefficient = 1.3414,
# R-squared = 0.9929, Adjusted R-squared = 0.9925.
# Python: Intercept = 0.0289, Market Return coefficient = 1.3414,
# R-squared = 0.993, Adjusted R-squared = 0.992.
# The small differences shown are due to rounding.

# Conclusion and Reflection:
# Excel is easier and more visual for simple financial analyses and small datasets.
# However, Python provides more customization, automation, reproducibility,
# and scalability. It is also more appropriate for large datasets and advanced
# financial models. For quick and simple analyses, I would use Excel. For repeated,
# large, or more complex financial analyses, I would prefer Python.