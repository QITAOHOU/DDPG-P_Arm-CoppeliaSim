# Introduction 
This repository contains my attempt to implement a Reinforcement Learning algorithm to a robotics scene in CoppeliaSim. Specifically, I am attempting to train a DDPG agent to pick up a box using a P_arm in a simulation. 
## Literature Review
Before attempting the problem, I did some background reading to identify what algorithm I wanted to use. I ended up settling on the DDPG as it was the most versatile and particularly applicable to continuous control problems like the dexterous robotics problem identified above.
Literature Review:
https://docs.google.com/document/d/1ftYiNFJFXYzdUeViU_1kHTteWqZeIcB9WCpcPf2pmME/edit?usp=sharing

## Set Up
I have split the project into two primary parts. Creating and interfacing with a CoppeliaSim scene, and implementing a DDPG algorithm.

### Coppelia Sim
I have set up a scene and am able to send torques to joint motors, which can be found under CoppeliaSim\Coppelia Scene. The scene is controlled by the python script "_main_.py" under CoppeliaSim\Python Programs. 

In order to run, make sure that you have a version of CoppeliaSim. Then, place the "b0.py" and "b0RemoteApi.py" and "_main_.py" files in the same folder as all the dependencies (found under CoppeliaSim\Python Programs\Dependancies). Open the CoppeliaSim scene called "First Attempt" from the Coppelia Scene folder, and once the BlueZero node has been created (should happen automatically), go to the "Add Ons" menu at the top and tick "b0RemoteApiServer". Leaving the scene open, run the Python Script. The robotic arm should execute a series of movements, and then the simulation ends.

### DDPG Agent



I am in the process of implementing the DDPG algorithm. I wanted to gain some experience with using TF Agents to implement the algorithm to get familiar with the various parts of a DDPG program,
and so am implementing a DDPG agent in the OpenAI Gym Minataur scene before I attempt to code up the CoppeliaSim environment for TF Agents.

Current Goals:

-Keep adjusting hyperparameters of Minatuar DDPG Agent for better convergence

-Code up CoppeliaSim environment for TF Agents

-Reimplement DDPG using just Tensorflow
