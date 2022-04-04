import numpy as np
import tensorflow as tf
from datetime import datetime


class StdScalerLinearRegressionTF(tf.Module):
    def __init__(self, input_shape, **kwargs):
        super().__init__(**kwargs)
        self.mean = tf.Variable(np.zeros(input_shape), dtype="float64", name="mean")
        self.variance = tf.Variable(np.zeros(input_shape) + 1, dtype="float64", name="variance")
        self.w = tf.Variable(np.zeros(input_shape) + 1, dtype="float64", name="pesos")
        self.b = tf.Variable(1.0, dtype="float64", name="bias")

    @tf.function
    def __call__(self, x):
        norm_x = (x - self.mean) / self.variance
        return tf.reduce_sum(self.w * norm_x) + self.b

    def predict(self, x):
        return self(x)


@tf.function
def std_scaler(x, mean, variance):
    return (x - mean) / variance


@tf.function
def tf_mse(y_true, y_pred):
  sq_diff = tf.pow(y_true - y_pred, 2)
  tf.print(sq_diff)
  return tf.reduce_mean(sq_diff)


def testing_if_graph_was_built_correctly(tf_function, params):
    graph_output = tf_function(*params)
    tf.config.run_functions_eagerly(True)
    eagerly_output = tf_function(*params)
    tf.config.run_functions_eagerly(False)
    assert graph_output == eagerly_output


def logging_the_graph():
    # Set up logging.
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = "logs/func/%s" % stamp
    writer = tf.summary.create_file_writer(logdir)

    # Create a new model to get a fresh trace
    # Otherwise the summary will not see the graph.
    new_model = StdScalerLinearRegressionTF(5)

    # Bracket the function call with
    # tf.summary.trace_on() and tf.summary.trace_export().
    tf.summary.trace_on(graph=True)
    tf.profiler.experimental.start(logdir)
    # Call only one tf.function when tracing.
    z = print(new_model(tf.constant([np.zeros(5)])))
    with writer.as_default():
        tf.summary.trace_export(
            name="my_func_trace",
            step=0,
            profiler_outdir=logdir)


if __name__ == "__main__":
    input_shape = 5
    model = StdScalerLinearRegressionTF(input_shape)
    x = np.zeros(input_shape) + 1
    y = model.predict(x)
    print("Prediction:", y.numpy())
    f_params = (np.zeros(10), np.zeros(10) + 1)
    testing_if_graph_was_built_correctly(tf_mse, f_params)

    # This is the graph-generating output of AutoGraph.
    # print(tf.autograph.to_code(tf_mse))

    # This is the graph itself.
    print(tf_mse.get_concrete_function(tf.constant(np.zeros(3)), tf.constant(np.zeros(3))).graph.as_graph_def())

    # Plot the graph in tensorboard
    logging_the_graph()
