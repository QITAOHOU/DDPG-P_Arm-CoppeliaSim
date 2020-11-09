# Introduction 
This repository contains my attempt to implement a Reinforcement Learning algorithm to a robotics scene in CoppeliaSim. Specifically, I am attempting to train a DDPG agent to pick up a box using a P_arm in a simulation. I am a complete novice, but decided to just get stuck in and see what I could do.
## Literature Review
Before attempting the problem, I did quite a bit of background reading to identify what algorithm I wanted to use. I ended up settling on the DDPG as it was the most versatile and particularly applicable to continuous control problems like the dexterous robotics problem identified above.
First attempt at implementing a RL Algorithm to a Robotics project. 
I have split the project into two primary parts. Creating and interfacing with a CoppeliaSim scene, and implementing a DDPG algorithm.
I have set up a scene and am able to send torques to joint motors, which can be found under CoppeliaSim.
I am in the process of implementing the DDPG algorithm. I wanted to gain some experience with using TF Agents to implement the algorithm to get familiar with the various parts of a DDPG program,
and so am implementing a DDPG agent in the OpenAI Gym Minataur scene before I attempt to code up the CoppeliaSim environment for TF Agents.

Current Goals:

-Keep adjusting hyperparameters of Minatuar DDPG Agent for better convergence

-Code up CoppeliaSim environment for TF Agents

-Reimplement DDPG using just Tensorflow
