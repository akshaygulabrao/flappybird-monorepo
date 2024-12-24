# MIT License
#
# Copyright (c) 2020 Gabriel Nogueira (Talendar)
# Copyright (c) 2023 Martin Kubovcik
# Copyright (c) 2024 Akshay Gulabrao
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

"""
Simple handcrafted agent for Flappy Bird.
"""


def agent(obs):
    """
    Simple handcrafted agent for Flappy Bird.

    Args:
        obs (list): Observation from the environment.

    Returns:
        int: Action to take (0 for no action, 1 for flap).
    """
    pipe = 0
    if obs[0] < 5:
        pipe = 1
    x = obs[pipe * 3]
    bot = obs[pipe * 3 + 2]
    top = obs[pipe * 3 + 1]
    y_next = obs[-3] + obs[-2] + 24 + 1
    action = 0
    if 74 < x < 88 and obs[-3] - 45 >= top:
        action = 1
    elif y_next >= bot:
        action = 1
    return action
