import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats import skew, kurtosis
from scipy.stats.mstats import gmean
from qiskit_finance.applications.optimization import PortfolioOptimization
from qiskit_algorithms import NumPyEigensolver, NumPyMinimumEigensolver
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.result import QuasiDistribution


# Data Acqusition

optimization_criterion = 'cvar'
symbols = ['AAPL', 'MSFT', 'GOOG', 'AMZN']
start_date = '2018-01-01'
end_date = '2023-01-01'
data = yf.download(symbols, start=start_date, end=end_date)['Adj Close']
returns = data.pct_change().dropna()

# Define classical mean-variance optimization function
def classical_mean_variance(data):
    returns = np.log(data / data.shift(1))
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    # Objective function to minimize - portfolio volatility
    def objective(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    # Constraint: weights sum up to 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    # Bounds for weights: between 0 and 1
    bounds = tuple((0, 1) for _ in range(len(data.columns)))

    # Initial guess for weights: equal weights
    initial_weights = np.array([1.0 / len(data.columns)] * len(data.columns))

    # Perform optimization
    result = minimize(objective, initial_weights, bounds=bounds, constraints=constraints)

    # Extract optimal weights
    optimal_weights = result.x

    return optimal_weights

# Define tickers, start date, and end date
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']  # Example tickers
start_date = '2020-01-01'
end_date = '2022-01-01'


# Perform classical mean-variance optimization
optimal_weights = classical_mean_variance(data)

# Print results
print("Optimal Portfolio Weights:")
for ticker, weight in zip(tickers, optimal_weights):
    print(f"{ticker}: {weight:.4f}")

returns = data.pct_change()

# Calculate mu (mean)
mu = returns.mean()

# Calculate sigma (covariance)
sigma = returns.cov()

plt.imshow(sigma, interpolation="nearest")
plt.show()

num_assets=len(data.columns)
q = 0.5  # set risk factor
budget = num_assets // 2  # set budget
penalty = num_assets  # set parameter to scale the budget penalty term

portfolio = PortfolioOptimization(
    expected_returns=mu.values, covariances=sigma.values, risk_factor=q, budget=budget
)
qp = portfolio.to_quadratic_program()

def print_result(result):
    selection = result.x
    value = result.fval
    print("Optimal: selection {}, value {:.4f}".format(selection, value))

    eigenstate = result.min_eigen_solver_result.eigenstate
    probabilities = (
        eigenstate.binary_probabilities()
        if isinstance(eigenstate, QuasiDistribution)
        else {k: np.abs(v) ** 2 for k, v in eigenstate.to_dict().items()}
    )
    print("\n----------------- Full result ---------------------")
    print("selection\tvalue\t\tprobability")
    print("---------------------------------------------------")
    probabilities = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)

    for k, v in probabilities:
        x = np.array([int(i) for i in list(reversed(k))])
        value = portfolio.to_quadratic_program().objective.evaluate(x)
        print("%10s\t%.4f\t\t%.4f" % (x, value, v))
exact_mes = NumPyMinimumEigensolver()
exact_eigensolver = MinimumEigenOptimizer(exact_mes)

result = exact_eigensolver.solve(qp)

print_result(result)