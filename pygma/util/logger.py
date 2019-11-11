import tensorflow as tf


class Logger:
    """Class for logging events.

    Attributes:
      logdir: Directory where log to
    """

    def __init__(self, logdir):
        self.logdir = logdir
        self.writer = tf.summary.create_file_writer(logdir)

    def log_scalar(self, name, value, step):
        """Logs one scalar value.

        Args:
            name: Scalar value name, str
            value: Scalar value, any
            step: The number of current step, int
        """
        with self.writer.as_default():
            tf.summary.scalar(name, value, step=step)

    def flush(self):
        """Makes actual log."""
        self.writer.flush()
