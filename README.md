# Logistic Regression with Numpy
<br><br>
## Overview<br>
This repository contains a Python implementation of Logistic Regression using only the Numpy library. Logistic Regression is a fundamental algorithm in machine learning, primarily used for binary classification tasks, and this implementation serves as a minimalistic yet effective example using numpy for educational purposes.
<br><br>
## Purpose<br>
Logistic Regression allows for modeling the probability that an instance belongs to a particular class. In this context:
<br><br>
X: Represents the input matrix with multiple features.<br>
Y: Represents the binary target variable we aim to predict.<br>
Sigmoid Function: Transforms the linear combination of input features into probabilities.<br>
Gradient Descent: Optimizes the model to find the best-fitting parameters (theta) for accurate predictions.<br>
This implementation serves as an educational resource to understand the inner workings of logistic regression and the implementation nuances using only the Numpy library.
<br><br>
The primary purpose of this implementation is to showcase the core concepts of Logistic Regression, including hypothesis formulation, cost computation, gradient descent optimization, and model training. The code emphasizes simplicity and clarity to aid understanding.
<br><br>
## Code Highlights <br>
### Logistic Regression Class
<br><br>
![46-4](https://github.com/hisanusman/Logistic-Regression-through-Numpy/assets/101946933/8630c26e-6bc4-450c-9b3d-2eabea0ef247)


The `LogisticRegression` class encapsulates the entire process of logistic regression. Here's a breakdown of its key components:
<br><br>
- **Sigmoid Function**: Computes the logistic hypothesis, transforming the linear combination of input features into probabilities.
- **Cost Function**: Calculates the cross-entropy loss between predicted probabilities and actual binary labels.
- **Derivative Function**: Computes the derivative of the cost function with respect to the model parameters (theta).
- **Gradient Descent**: Updates model parameters using the calculated derivative and a specified learning rate.
- **Training**: Initializes random theta values and iteratively performs gradient descent to optimize the model.
- **Prediction**: Uses the trained model to make binary predictions on new data.
- **Predict Confidence**: Returns the predicted probabilities for each class.
- **Get Weights**: Returns the flattened theta values.
<br><br>
### Code Structure <br>
The code is structured to be modular and comprehensible. Each function is well-defined with comments to facilitate understanding. The structure is designed to encourage exploration and modification for learning purposes.
<br><br>
## Usage <br>
To use the logistic regression model:

```python
# Example Usage

# Initialize the model
model = LogisticRegression()

# Train the model
model.train(X, Y, lr=0.0001, num_iters=1000, num_printing=100)

# Make binary predictions
predictions = model.predict(X_test)

# Get predicted probabilities
probabilities = model.predict_confidence(X_test)

# Get model weights
weights = model.get_weights()
```
<br><br>
Feel free to reach out for any questions, suggestions, or improvements!
