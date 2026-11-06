import math
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

# x_{k+1} = α·x_k − (1 − α)·K·a_{k−d}, with α = e^(−Δt/τ) (discretization)
# ẋ(t) = −(1/τ)·x(t) − (K/τ)·a(t − T)
# y_d(t) = x(t) output is equal to the state

# where
# d is a step delay
# Δt is a time-step
# T is a delay in second
# tau is a system lag in second
# a is a user input

# system parameters
l = 22  # lowest temperature in celcius
h = 42  # highest temperature in celcius
Temp_mid = (h + l) / 2  #
K = (h - l) / math.pi  # celcius per rad (user can rotate knod -pi/2 to pi/2)
T = 1  # system delay (sec)
tau = 0.5  # system lag (sec)
dt = 0.01  # simulation time step
d = math.round(T / dt)  # number of delay steps
alpha = np.exp(-(dt / tau))

# number of simulation steps
k_max = 1000
# mock user input
steps = np.linspace(0, k_max - 1, k_max, dtype=int)
# a_user = np.zeros(k_max)
# a_user = np.full(k_max, 0.5)
a_user = np.cos(0.02 * steps) + np.sin(0.05 * steps)
a_user = np.clip(a_user, -math.pi / 2, math.pi / 2)
# inital input and state
a_init = a_user[0]  # intial knop angle (rad)
x_init = -K * a_init

# buffer
a_buffer = deque(maxlen=d)  # buffer to queue user input
a_buffer.extend(np.full(d, a_init))  # intial user's input
state_history = np.zeros(k_max)
output_history = np.zeros(k_max)


def step(x_k, a_k_d):
    x_next = (alpha * x_k) - ((1 - alpha) * K * a_k_d)
    return x_next


def run(x_init, a_user):
    x_k = x_init

    for i in steps:
        state_history[i] = x_k
        output_history[i] = Temp_mid + x_k

        # run step
        a_k_d = a_buffer.popleft()
        x_next = step(x_k=x_k, a_k_d=a_k_d)

        # update
        x_k = x_next
        a_buffer.append(a_user[i])


if __name__ == "__main__":
    run(x_init=x_init, input=input)

    t = steps * dt
    plt.plot(t, Temp_mid - K * input, label="Requested temperature")
    plt.plot(t, output_history, label="Shower temperature")
    plt.axhline(l, ls="--", c="gray")
    plt.axhline(h, ls="--", c="gray")
    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.show()
