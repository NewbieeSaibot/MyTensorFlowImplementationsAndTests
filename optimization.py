import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from joblib import load


class LinearRegressionTF(tf.Module):
    def __init__(self, sklearn_model, **kwargs):
        super().__init__(**kwargs)
        self.mean = tf.Variable(np.zeros(len(sklearn_model.coef_)), name="mean")
        self.variance = tf.Variable(np.zeros(len(sklearn_model.coef_)) + 1, name="variance")
        self.w = tf.Variable(sklearn_model.coef_, name="pesos")
        self.b = tf.Variable(sklearn_model.intercept_, name="bias")

    def __call__(self, x):
        norm_x = (x - self.mean) / self.variance
        return tf.reduce_sum(self.w * norm_x) + self.b


class TensorFlowSolver(tf.Module):
    def __init__(self, ranges, **kwargs):
        super().__init__(**kwargs)
        self.opt_vars = None
        self.ranges = ranges
        self.initialize_opt_vars()

    def initialize_opt_vars(self):
        initial_vars = []
        for i in range(len(self.ranges)):
            initial_vars.append(sum(self.ranges[i])/2)
        self.opt_vars = tf.Variable(initial_vars, name="opt_vars", dtype="float64")

    def optimize(self, objective_function, learning_rate, n_iterations, verbose=True):
        for i in range(n_iterations):
            with tf.GradientTape() as t:
                # Trainable variables are automatically tracked by GradientTape
                current_loss = objective_function(self.opt_vars)

            # Use GradientTape to calculate the gradients with respect to W and b
            print(current_loss)
            diffs = t.gradient(current_loss, self.opt_vars)

            # Subtract the gradient scaled by the learning rate
            self.opt_vars.assign_sub(learning_rate * diffs)
            # Check for variables out of limit
            for i in range(len(self.ranges)):
                if self.ranges[i][0] > self.opt_vars[i]:
                    self.opt_vars[i].assign(self.ranges[i][0])
                if self.ranges[i][1] < self.opt_vars[i]:
                    self.opt_vars[i].assign(self.ranges[i][1])

            if verbose:
                print("Obj:", current_loss.numpy())
                print("Variables:", self.opt_vars.numpy())

        if verbose:
            print([var.name for var in t.watched_variables()])


linreg = LinearRegression()
x1 = np.random.randint(0, 100, 50)
x2 = np.random.randint(0, 100, 50)
y = 2 * x1 - 3 * x2 + 5

df = pd.DataFrame()
df['x1'] = x1
df['x2'] = x2
df['y'] = y

linreg.fit(df[['x1', 'x2']], df['y'])
lr_tf = LinearRegressionTF(linreg)
print("previsao sklearn", linreg.predict([[0, 0]]))
print("previsão tf", lr_tf([0, 0]))

print("Multiplicação", tf.Variable([5, 5]) * tf.Variable([1, 2]))


def obj_function(x):
    return lr_tf(x)


optm = TensorFlowSolver([[0, 5], [0, 3]])
optm.optimize(obj_function, 0.2, 100)
print(optm.opt_vars.numpy())

pipe = load(filename="./U11_TX7001_Ridge_U_2A")
print(pipe[0].mean_, pipe[0].var_)
