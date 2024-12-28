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


def handcrafted_agent(obs,normalize=True):
    if not normalize:
        pipe = 0
        if obs[0] < 5:
            pipe = 1
        x = obs[pipe *3]
        bot = obs[pipe * 3 + 2]
        top = obs[pipe * 3 + 1]
        y_next = obs[-2] + obs[-1] + 24 + 1
        if 74 < x < 88 and obs[-2] - 45 >= top:
            return 1
        elif y_next >= bot:
            return 1
        return 0
    else:
        pipe = 0
        if obs[0] < 5/288:
            pipe = 1
        x = obs[pipe *3]
        bot = obs[pipe * 3 + 2]
        top = obs[pipe * 3 + 1]
   
        y_next = obs[-2] + (obs[-1]*10) / 512 # current y + current y_velocity
        y_next += 1/512 # 1 pixel acceleration per frame
        y_next += 24/512 # height of bird
        if 72/288 < x < 88/288 and obs[-2] - 45/512 >= top:
                return 1
        elif y_next >= bot:
            return 1
        return 0