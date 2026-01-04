import numpy as np
import matplotlib.pyplot as plt
import math

# Linear Regression using Gradient Descent
# y = mx + b
y = np.array([3,6,4,8,9,12,16])
X = np.array([5,6,7,8,9,10,11])
m = 0
b = 0
alpha = 0.1 
iteration = 1000
n = len(X)
for i in range(iteration):
    J = (1/n)*sum(((m*X+b)-y)**2)
    m = m- alpha*((2/n)*sum((m*X+b)-y))
    b = b- alpha*((2/n)*sum((m*X+b)-y))
    print('cost {} in {} iteration'.format(J,i))
print('m {} b {}'.format(m,b))

# Gradient Descent Visualization for Linear Regression

n = 5
a = 0
b = 0
learning_rate = 0.01
x = np.array([1,2,3,4,5])
y = np.array([5,8,11,14,17])

plt.ion()  # live updating

for i in range(100):
    y_pred = a*x + b
    
    # Compute cost
    cost = (1/n) * np.sum((y - y_pred)**2)

    plt.scatter(a, cost)   # cost vs a
    plt.pause(0.01)        # needed to update during loop

    # Compute gradients
    da = -(2/n) * np.sum(x * (y - y_pred)) 
    db = -(2/n) * np.sum(y - y_pred)

    # Update parameters
    a = a - learning_rate * da
    b = b - learning_rate * db

    if math.isclose(cost, 0.0, abs_tol=1e-9):
        break

plt.ioff()
plt.show()
