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


class BasePolicy:
    """Base class for all policies of the package."""

    def get_action(self, obs):
        """Returns action for specific observation.

        Args:
            obs: observation of the environment.

        Returns:
            An action which is recommended by policy.    
        """
        raise NotImplementedError

    def update(self, acs, obs):
        """Updates policy.

        Args:
            acs: Actions
            obs: Observations
        """
        raise NotImplementedError

    def save(self, filename):
        raise NotImplementedError

    def restore(self, filename):
        raise NotImplementedError

    def test(self, a, b):
        """Test docstring.

        .. warning::
            This function is intended as a template for documenting the code.

        Args:
            a (int): A
            b (float): B

        Returns:
            bool: result
        """
        pass
