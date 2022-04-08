import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class MyNonLinearModel(tf.Module):
    """
    Model function: F(x) = c0 * feature_1 ^ exponent + bias
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bias = tf.Variable(1.0, dtype="float64", name="bias", trainable=True)
        self.c0 = tf.Variable(1.0, dtype="float64", name="c0", trainable=True)
        self.exponent = tf.Variable(1.0, dtype="float64", name="exponent", trainable=True)

    def __call__(self, x):
        result = self.c0 * tf.pow(x, self.exponent) + self.bias
        return result

    def predict(self, x):
        return self(x)

    def train(self, x, y, learning_rate, epochs):
        optm = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        for epoch in range(epochs):
            with tf.GradientTape() as t:
                loss = self.loss(y, self(x))

            # self.variables returns all variables
            # self.trainable_variables returns all trainable variable.
            # This can be useful to apply gradients to all trainable variables faster.
            # Use GradientTape to calculate the gradients with respect to W and b
            grads = t.gradient(loss, self.trainable_variables)
            tf.print(f"Grads at epoch {epoch}:", grads)
            # Subtract the gradient scaled by the learning rate
            optm.apply_gradients(zip(grads, self.trainable_variables))

            # The above method is easier because it already applies the gradients for all trainable variables
            # Another option would be the following code:
            # for i in range(len(self.trainable_variables)):
            #     self.trainable_variables[i].assign_sub(learning_rate * grads[i])

    @staticmethod
    def loss(target_y, predicted_y):
        # Observe that the inputs are vector.
        # The reduce mean will make it return a single value for all the batch.
        return tf.reduce_mean(tf.square(target_y - predicted_y))


def data_generator(n_samples=300, c0=10, bias=5, exponent=1.8, noise_amp=100):
    """
    Model function: F(x) = c0 * feature_1 ^ exponent + bias + noise
    """
    np.random.seed(1)
    noise = np.random.normal(0, noise_amp, n_samples)  # No DC level
    x = np.random.uniform(1, 10, n_samples)
    y = c0 * x ** exponent + bias + noise
    under_y = c0 * x ** exponent + bias
    return x, y, under_y


if __name__ == "__main__":
    x, y, u_y = data_generator()
    model = MyNonLinearModel()
    print("before train variables:")
    for var in model.trainable_variables:
        print(var.name, var.numpy())

    model.train(x, y, 0.1, 200)
    predicted_y = model.predict(x)

    print("trained variables:")
    for var in model.trainable_variables:
        print(var.name, var.numpy())

    # Plotting real vs. predicted
    fig, ax = plt.subplots()
    ax.plot(x, y, '.', alpha=0.5, label="Curve with noise")
    ax.plot(x, predicted_y, '.', alpha=0.5, label="Detected curve")
    ax.plot(x, u_y, '.', alpha=0.5, label="Underlying curve")

    ax.legend()
    plt.show()
