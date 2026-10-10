# linear-regression-from-scratch
In this repository, I explore different types of linear regression and hand-derive the math needed for them from scratch. So far, I have explored gradient descent, but there is more to come.

# linear-regression-from-scratch

Implementing linear regression from first principles: hand-deriving the
math, coding the algorithms in pure NumPy, and verifying against
scikit-learn. Part of a self-directed AI/ML study plan.

---

## Linear Regression — Theory Notes

Notes from working through the model: what is being fit, what the
residuals represent, and why we minimize squared error.

![Linear regression notes, page 1](linreg1.png)
![Linear regression notes, page 2](linreg2.png)
![Linear regression notes, page 3](linreg3.png)

---

## Deriving the MSE Gradients

The loss function:

$$L = \frac{1}{n}\sum_{i=1}^{n}\left(y_i - (mx_i + b)\right)^2$$

Hand-derived partial derivatives with respect to the slope and intercept:

![MSE derivation, page 1](msederiv1.png)
![MSE derivation, page 2](msederiv2.png)

Results:

$$\frac{\partial L}{\partial m} = -\frac{2}{n}\sum_{i=1}^{n} x_i\left(y_i - \hat{y}_i\right)
\qquad
\frac{\partial L}{\partial b} = -\frac{2}{n}\sum_{i=1}^{n}\left(y_i - \hat{y}_i\right)$$

These two expressions are what gradient descent evaluates at each step to
decide which direction to move the parameters.

---

## Implementation

`Gradient Descent From Scratch.ipynb` contains a `LinearRegressionGd` class
written in pure NumPy (no scikit-learn):

- `__init__` — stores the learning rate and epoch count, initializes m and b to 0
- `predict` — applies the current line to the input
- `fit` — runs the training loop: predict, compute residuals, record MSE,
  evaluate both gradients, step both parameters against them

Tested on synthetic data generated as `y = 3x + 7 + noise`, so the true
parameters are known in advance. The model recovers m ≈ 3.0 and b ≈ 7.0.
