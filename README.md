# Shower Simulation
This repository simulates a simple shower model with two behaviors:
1. Time delay ($T$)
2. System lag ($\tau$)

Imagine turning the knob to adjust the water temperature: the temperature does not change immediately. This is because of the <u>delay</u> while the water travels along the pipe, and <u>the lag</u> caused by water at different temperatures mixing together.

## Shower Model
### State Equation
First, we model the shower system as a continuous-time equation:
$$
\dot{x}(t) = -\frac{1}{\tau}x(t) - \frac{K}{\tau}a(t-T)
$$

$$
y(t) = x(t)
$$

where<br>
$T$ is the time delay (s)<br>
$\tau$ is the system lag (s)<br>
$a$ is the user input: the angle the user rotates the faucet knob (rad)<br>
$K$ is the gain, i.e., the temperature change per unit of knob angle ($^\circ\text{C}$/rad)<br>
$x$ and $y$ are the state and the output, respectively. Both represent the water temperature ($^\circ\text{C}$).

### Discretization
Next, we discretize the equation so that it can be implemented in code:
$$
x[k+1] = \alpha x[k] - (1-\alpha)Ka[k-d]
$$
$$
y[k] = x[k]
$$
where<br>
$\alpha = e^{-\frac{\Delta t}{\tau}}$<br>
$\Delta t$ is the simulation time step (s)<br>
$d$ is the number of delay steps ($= \frac{T}{\Delta t}$)<br>

## Repository Structure
This repository contains two main files:
1. `test_model.py`: tests the correctness of the model and shows the system response to different inputs.
2. `shower_demo.py`: uses the model to create an interactive simulation where you can change the input and see the response in real time. The simulation is built with the `arcade` package, so please install the required packages listed in `requirements.txt` before running this file.