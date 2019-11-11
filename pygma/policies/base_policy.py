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
"""Class to represent BasePolicy.

BasePolicy is used to define a typical API to policy instances,
used throughout the package.
"""

from abc import ABC, abstractmethod


class BasePolicy(ABC):
    """Base class for pigma's `policy` concept."""

    def __init__(self, **kwargs):
        super(BasePolicy, self).__init__(**kwargs)

    @abstractmethod
    def get_action(self, obs):
        """Returns action for specific observation.

        Args:
            obs: observation of the environment.

        Returns:
            An action which is recommended by policy.    
        """
        pass

    @abstractmethod
    def save(self, filename):
        """Saves policy to disc.

        Args:
            filename: File to save policy to.
        """
        pass

    @abstractmethod
    def restore(self, filename):
        """Restores policy from file.

        Args:
            filename: File wit saved policy.
        """
        pass
