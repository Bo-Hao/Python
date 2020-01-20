import tensorflow as tf 
import numpy as np 
import torch 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 


learning_rate = 0.1
n_samples = 200
training_steps = 1000
display_step = 10
#Create dataset
x = torch.unsqueeze(torch.linspace(-1, 1, n_samples), dim=1)  # x data (tensor), shape=(100, 1)
y = x.mul(2) +3+0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)

x = np.array(list(x))
y = np.array(list(y))

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

X = X_train
Y = y_train
W = tf.Variable(tf.random.normal([1]))
b = tf.Variable(tf.zeros([1]))

#Define LR and Loss function (MSE)
def linear_regression(x):
    return W * x + b
  
def mean_square(y_pred, y_true):
    return tf.reduce_sum(tf.pow(y_pred-y_true, 2)) / (2 * n_samples)

# Stochastic Gradient Descent Optimizer.
optimizer = tf.optimizers.Adam(learning_rate)

def run_optimization():
    # Wrap computation inside a GradientTape for automatic differentiation.
    with tf.GradientTape() as g:
        pred = linear_regression(X)
        
        loss = mean_square(pred, Y)

    # Compute gradients.
    gradients = g.gradient(loss, [b, W])
    
    # Update W and b following gradients.
    optimizer.apply_gradients(zip(gradients, [b, W]))


for step in range(1, training_steps + 1):
    # Run the optimization to update W and b values.
    run_optimization() 
    if step % display_step == 0:
        pred = linear_regression(X)
        loss = mean_square(pred, Y)
        print("step: %i, loss: %f, W: %f, b: %f" % (step, loss, W.numpy(), b.numpy()))



plt.scatter(X_train, y_train, c = 'green')
x = np.array([X_test])
plt.scatter(x, linear_regression(x), c = 'red')

plt.show()