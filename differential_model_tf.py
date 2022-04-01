import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# Create model class
class MyModel(tf.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize the weights to `5.0` and the bias to `0.0`
        # In practice, these should be randomly initialized
        self.w = tf.Variable(5.0)
        self.b = tf.Variable(0.0)
        self.actual_resistance = tf.Variable(0.0)

    def __call__(self, x):  # x = [u_var, u_var, cleaning_rate, cleaning_rate]
        u_maximo = self.w * x + self.b
        cleaning_rate = x * x
        self.actual_resistance = self.actual_resistance + cleaning_rate
        y = u_maximo - self.actual_resistance
        return y


def generate_data(true_w, true_b, num_examples):
    # A vector of random x values
    x = tf.linspace(1, 4, num_examples)
    x = tf.cast(x, tf.float32)

    def f(x):
        return x ** 3 * true_w + true_b

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
        current_loss = loss(y, model(x))

    # Use GradientTape to calculate the gradients with respect to W and b
    dw, db, dexp = t.gradient(current_loss, [model.w, model.b, model.expoente])
    print("Gradientes:", dw, db, dexp)
    # Subtract the gradient scaled by the learning rate
    model.w.assign_sub(learning_rate * dw)
    model.b.assign_sub(learning_rate * db)
    model.expoente.assign_sub(learning_rate * dexp)


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
    print("model response", model(3.0), model(3.0).numpy())
    weights = []
    biases = []
    epochs = range(100)

    def report(model, loss):
        return f"W = {model.w.numpy():1.2f}, b = {model.b.numpy():1.2f}, exp = {model.expoente.numpy():1.2f}, loss={loss:2.5f}"

    for epoch in epochs:
        # Update the model with the single giant batch
        train(model, x, y, learning_rate=0.00001)

        # Track this before I update
        weights.append(model.w.numpy())
        biases.append(model.b.numpy())
        current_loss = loss(y, model(x))

        print(f"Epoch {epoch:2d}:")
        print("    ", report(model, current_loss))


if __name__ == '__main__':
    linear_regression()
