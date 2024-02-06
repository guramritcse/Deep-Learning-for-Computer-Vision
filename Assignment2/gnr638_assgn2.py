import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def forward_pass(X, W1, W2):
    A1 = np.dot(X, W1)
    Z1 = sigmoid(A1)
    A2 = np.dot(Z1, W2)
    Z2 = sigmoid(A2)
    return A1, Z1, A2, Z2

def calculate_loss(Z2, Y):
    return 0.5 * np.sum((Z2 - Y) ** 2)

def calculate_mse(weights1, weights2):
    return np.mean((weights1 - weights2) ** 2)

def backward_pass(X, Z1, A2, Z2, Y, W2):
    m = X.shape[0]

    dZ2 = Z2 - Y
    dA2 = dZ2 * sigmoid_derivative(A2)
    dW2 = np.dot(Z1.T, dA2)

    dZ1 = np.dot(dA2, W2.T)
    dA1 = dZ1 * sigmoid_derivative(A1)
    dW1 = np.dot(X.T, dA1)

    return dW1, dW2

def numerical_gradient_check(X, Y, W1, W2, epsilon=1e-5):
    gradients = []

    for weight_matrix, gradient_matrix in zip([W1, W2], [np.zeros_like(W1), np.zeros_like(W2)]):
        rows, cols = weight_matrix.shape
        # print(weight_matrix , gradient_matrix)
        for i in range(rows):
            for j in range(cols):
                # Perturb the weight matrix
                weight_matrix[i, j] += epsilon

                # Forward pass with perturbed weight matrix
                _, _, _, Z2_plus_epsilon = forward_pass(X, W1, W2)
                loss_plus_epsilon = calculate_loss(Z2_plus_epsilon, Y)

                # Reset the weight matrix to original value
                weight_matrix[i, j] -= 2 * epsilon

                # Forward pass with perturbed weight matrix
                _, _, _, Z2_minus_epsilon = forward_pass(X, W1, W2)
                loss_minus_epsilon = calculate_loss(Z2_minus_epsilon, Y)

                # Numerical gradient approximation
                numerical_gradient = (loss_plus_epsilon - loss_minus_epsilon) / (2 * epsilon)

                # Reset the weight matrix to original value
                weight_matrix[i, j] += epsilon

                gradient_matrix[i, j] = numerical_gradient

        gradients.append(gradient_matrix)

    return gradients

# Example usage
X = np.array([[1, 1]])
Y = np.array([[0, 1]])

# Initialize weights
np.random.seed(42)
W1 = np.random.rand(2, 2)
W2 = np.random.rand(2, 2)


epsilon_values = np.logspace(-10, 0, 11)  # Vary epsilon from 10^(-10) to 1
accuracy_values = []

for epsilon in epsilon_values:
    # Forward pass
    A1, Z1, A2, Z2 = forward_pass(X, W1, W2)

    # Backward pass
    dW1, dW2 = backward_pass(X, Z1, A2, Z2, Y, W2)

    # Numerical gradient check
    numerical_gradients = numerical_gradient_check(X, Y, W1, W2, epsilon)

    # Calculate accuracy
    # accuracy = np.allclose(dW1, numerical_gradients[0]) and np.allclose(dW2, numerical_gradients[1])
     # Calculate MSE
    mse_w1 = calculate_mse(dW1, numerical_gradients[0])
    mse_w2 = calculate_mse(dW2, numerical_gradients[1])
    accuracy_values.append(np.log((mse_w1 + mse_w2) / 2))
    print(np.log((mse_w1 + mse_w2) / 2))
    # accuracy_values.append(mse_values)

# Plotting
plt.figure(figsize=(10, 6))
plt.semilogx(epsilon_values, accuracy_values, marker='o', linestyle='-')
plt.title('Numerical Gradient Error vs. Epsilon')
plt.xlabel('Epsilon')
plt.ylabel('Log of Error')
plt.ylim(-60, -10) 
plt.grid(True)
plt.show()

# Print gradients for comparison
print("Analytical Gradients:")
print(dW1)
print(dW2)

print("\nNumerical Gradients:")
print(numerical_gradients[0])
print(numerical_gradients[1])
