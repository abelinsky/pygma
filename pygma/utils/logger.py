# Copyright (c) 2019 Alexander Belinsky

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# ==============================================================================
"""Utilities for logging to tensorboard."""
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
