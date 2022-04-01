import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# Create model class
class MyModel(tf.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize the weights to `5.0` and the bias to `0.0`
        # In practice, these should be randomly initialized
        self.p = tf.Variable([5.0, 0.0])

    def __call__(self, x, funcao):
        return funcao(self.p[0], x, self.p[1])


def funcao(a, x, b):
    return a * x + b


def generate_data(true_w, true_b, num_examples):
    # A vector of random x values
    x = tf.linspace(-2, 2, num_examples)
    x = tf.cast(x, tf.float32)

    def f(x):
        return x * true_w + true_b

    # Generate some noise
    noise = tf.random.normal(shape=[num_examples])

    # Calculate y
    y = f(x) + noise
    return x, y


def plot_data(x, y):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, y, '.')
    plt.show()


# Given a callable model, inputs, outputs, and a learning rate...
def train(model, x, y, learning_rate):
    with tf.GradientTape() as t:
        # Trainable variables are automatically tracked by GradientTape
        current_loss = loss(y, model(x, funcao))

    # Use GradientTape to calculate the gradients with respect to W and b
    dw = t.gradient(current_loss, model.p)

    # Subtract the gradient scaled by the learning rate
    model.p.assign_sub(learning_rate * dw)


# This computes a single loss value for an entire batch
def loss(target_y, predicted_y):
    return tf.reduce_mean(tf.square(target_y - predicted_y))


def linear_regression():
    # The actual line
    # Get Data
    TRUE_W = 3.0
    TRUE_B = 2.0
    NUM_EXAMPLES = 201
    x, y = generate_data(TRUE_W, TRUE_B, NUM_EXAMPLES)

    # Plot data
    plot_data(x, y)

    model = MyModel()
    # List the variables tf.modules's built-in variable aggregation.
    print("Variables:", model.variables)

    # Verify the model works
    # assert model(3.0).numpy() == 15.0
    # print("model response", model(3.0), model(3.0).numpy())
    weights = []
    biases = []
    epochs = range(100)

    # def report(model, loss):
    #     return f"W = {model.p.numpy():1.2f}, b = {model.p.numpy():1.2f}, loss={loss:2.5f}"

    for epoch in epochs:
        # Update the model with the single giant batch
        train(model, x, y, learning_rate=0.1)

        # Track this before I update
        weights.append(model.p.numpy())
        biases.append(model.p.numpy())
        current_loss = loss(y, model(x, funcao))

        print(f"Epoch {epoch:2d}:")
        print("Current Loss:", current_loss)


if __name__ == '__main__':
    linear_regression()
