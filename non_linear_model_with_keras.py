import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# Create model class
class MyModelKeras(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize the weights to `5.0` and the bias to `0.0`
        # In practice, these should be randomly initialized
        self.exp = tf.Variable(2.0)
        self.w = tf.Variable(5.0)
        self.b = tf.Variable(0.0)

    def call(self, x):
        return self.w * x ** self.exp + self.b


def generate_data(true_w, true_b, true_exp, num_examples):
    # A vector of random x values
    x = tf.linspace(0.5, 4.5, num_examples)
    x = tf.cast(x, tf.float32)

    def f(x):
        return x ** true_exp * true_w + true_b

    # Generate some noise
    noise = tf.random.normal(shape=[num_examples])

    # Calculate y
    y = f(x) + noise
    return x, y


def plot_data(x, y):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, y, '.')
    plt.show()


# This computes a single loss value for an entire batch
def loss(target_y, predicted_y):
    return tf.reduce_mean(tf.square(target_y - predicted_y))


def non_linear_regression():
    # The actual line
    # Get Data
    TRUE_W = 3.0
    TRUE_B = 2.0
    TRUE_EXP = 3.0
    NUM_EXAMPLES = 201
    x, y = generate_data(TRUE_W, TRUE_B, TRUE_EXP, NUM_EXAMPLES)
    print(x, y)
    # Plot data
    # plot_data(x, y)

    model = MyModelKeras()
    model.compile(
        # By default, fit() uses tf.function().  You can
        # turn that off for debugging, but it is on now.
        run_eagerly=False,

        # Using a built-in optimizer, configuring as an object
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),

        # Keras comes with built-in MSE error
        # However, you could use the loss function
        # defined above
        loss=loss,
    )
    print(model(3.0))

    # model.fit(x, y, epochs=100, batch_size=10)
    # print(model.exp.numpy(), model.b.numpy(), model.w.numpy())


if __name__ == '__main__':
    # non_linear_regression()
    x = tf.constant([[1., 1.], [3.14, 2.], [4., 5.]])
    print(tf.sin(x))
