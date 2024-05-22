# Programming Assignment: Reinforcement Learning

In the final graded practice lab of this course (and this specialization), you'll implement code for a virtual lunar lander. 
This will give you a chance to practice some of the concepts in deep reinforcement learning covered in lecture.  At the end of the lab, 
you'll see a video of your lunar lander and can check whether it has learned to successfully land on the moon's surface!

## 1 - Import Packages

We'll make use of the following packages:
- `numpy` is a package for scientific computing in python.
- `deque` will be our data structure for our memory buffer.
- `namedtuple` will be used to store the experience tuples.
- The `gym` toolkit is a collection of environments that can be used to test reinforcement learning algorithms. We should note that in this notebook we are using `gym` version `0.24.0`.
- `PIL.Image` and `pyvirtualdisplay` are needed to render the Lunar Lander environment.
- We will use several modules from the `tensorflow.keras` framework for building deep learning models.
- `utils` is a module that contains helper functions for this assignment. You do not need to modify the code in this file.

Run the cell below to import all the necessary packages.
```
import time
from collections import deque, namedtuple

import gym
import numpy as np
import PIL.Image
import tensorflow as tf
import utils

from pyvirtualdisplay import Display
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import Adam
```
## 2 - Hyperparameters

Run the cell below to set the hyperparameters.
```
MEMORY_SIZE = 100_000     # size of memory buffer
GAMMA = 0.995             # discount factor
ALPHA = 1e-3              # learning rate  
NUM_STEPS_FOR_UPDATE = 4  # perform a learning update every C time steps
```
## 3 - The Lunar Lander Environment

In this notebook we will be using [OpenAI's Gym Library](https://www.gymlibrary.dev/). The Gym library provides a wide variety of 
environments for reinforcement learning. To put it simply, an environment represents a problem or task to be solved. In this notebook, 
we will try to solve the Lunar Lander environment using reinforcement learning.

The goal of the Lunar Lander environment is to land the lunar lander safely on the landing pad on the surface of the moon. The 
landing pad is designated by two flag poles and its center is at coordinates `(0,0)` but the lander is also allowed to land outside 
of the landing pad. The lander starts at the top center of the environment with a random initial force applied to its center of mass 
and has infinite fuel. The environment is considered solved if you get `200` points. 

### 3.1 Action Space

The agent has four discrete actions available:

* Do nothing.
* Fire right engine.
* Fire main engine.
* Fire left engine.

Each action has a corresponding numerical value:

```
Do nothing = 0
Fire right engine = 1
Fire main engine = 2
Fire left engine = 3
```

### 3.2 Observation Space

The agent's observation space consists of a state vector with 8 variables:

* Its `(x,y)` coordinates. The landing pad is always at coordinates `(0,0)`.
* Its linear velocities `(\dot x,\dot y)`.
* Its angle `\theta`.
* Its angular velocity `\dot \theta`.
* Two booleans, `l` and `r`, that represent whether each leg is in contact with the ground or not.

### 3.3 Rewards

After every step, a reward is granted. The total reward of an episode is the sum of the rewards for all the steps within that episode.

For each step, the reward:
- is increased/decreased the closer/further the lander is to the landing pad.
- is increased/decreased the slower/faster the lander is moving.
- is decreased the more the lander is tilted (angle not horizontal).
- is increased by 10 points for each leg that is in contact with the ground.
- is decreased by 0.03 points each frame a side engine is firing.
- is decreased by 0.3 points each frame the main engine is firing.

The episode receives an additional reward of -100 or +100 points for crashing or landing safely respectively.

### 3.4 Episode Termination

An episode ends (i.e the environment enters a terminal state) if:

* The lunar lander crashes (i.e if the body of the lunar lander comes in contact with the surface of the moon).

* The absolute value of the lander's `x`-coordinate is greater than 1 (i.e. it goes beyond the left or right border)

You can check out the [Open AI Gym documentation](https://www.gymlibrary.dev/environments/box2d/lunar_lander/) for a full description of the environment. 

## 4 - Load the Environment

We start by loading the `LunarLander-v2` environment from the `gym` library by using the `.make()` method. `LunarLander-v2` is the latest version of the 
Lunar Lander environment and you can read about its version history in the [Open AI Gym documentation](https://www.gymlibrary.dev/environments/box2d/lunar_lander/#version-history).
```
env = gym.make('LunarLander-v2')
```
Once we load the environment we use the `.reset()` method to reset the environment to the initial state. The lander starts at the top center of the environment 
and we can render the first frame of the environment by using the `.render()` method.
```
env.reset()
PIL.Image.fromarray(env.render(mode='rgb_array'))
```
In order to build our neural network later on we need to know the size of the state vector and the number of valid actions. We can get this information from our environment by using the `.observation_space.shape` and `action_space.n` methods, respectively.
```
state_size = env.observation_space.shape
num_actions = env.action_space.n

print('State Shape:', state_size)
print('Number of actions:', num_actions)
```
