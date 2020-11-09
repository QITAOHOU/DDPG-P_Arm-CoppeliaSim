# DDPG-P_Arm-CoppeliaSim
First attempt at implementing a RL Algorithm to a Robotics project. 
I have split the project into two primary parts. Creating and interfacing with a CoppeliaSim scene, and implementing a DDPG algorithm.
I have set up a scene and am able to send torques to joint motors, which can be found under CoppeliaSim.
I am in the process of implementing the DDPG algorithm. I wanted to gain some experience with using TF Agents to implement the algorithm to get familiar with the various parts of a DDPG program,
and so am implementing a DDPG agent in the OpenAI Gym Minataur scene before I attempt to code up the CoppeliaSim environment for TF Agents.

Current Goals:

-Keep adjusting hyperparameters of Minatuar DDPG Agent for better convergence

-Code up CoppeliaSim environment for TF Agents

-Reimplement DDPG using just Tensorflow
