## Introduction 
This repository contains my attempt to implement a Reinforcement Learning algorithm to a robotics scene in CoppeliaSim. Specifically, I am attempting to train a DDPG agent to pick up a box using a P_arm in a simulation. 
## Literature Review
Before attempting the problem, I did some background reading to identify what algorithm I wanted to use. I ended up settling on the DDPG as it was the most versatile and particularly applicable to continuous control problems like the dexterous robotics problem identified above.
Literature Review:
https://docs.google.com/document/d/1ftYiNFJFXYzdUeViU_1kHTteWqZeIcB9WCpcPf2pmME/edit?usp=sharing

## Set Up
I have split the project into two primary parts. Creating and interfacing with a CoppeliaSim scene, and implementing a DDPG algorithm.

#### Coppelia Sim
I have set up a scene and am able to send torques to joint motors, which can be found under CoppeliaSim\Coppelia Scene. The scene is controlled by the python script "_main_.py" under CoppeliaSim\Python Programs. 

In order to run, make sure that you have a version of CoppeliaSim. Then, place the "b0.py" and "b0RemoteApi.py" and "_main_.py" files in the same folder as all the dependencies (found under CoppeliaSim\Python Programs\Dependancies). Open the CoppeliaSim scene called "First Attempt" from the Coppelia Scene folder, and once the BlueZero node has been created (should happen automatically), go to the "Add Ons" menu at the top and tick "b0RemoteApiServer". Leaving the scene open, run the Python Script. The robotic arm should execute a series of movements, and then end the simulation.

#### DDPG Agent
After having familiarized myself with CoppeliaSim, I went about implementing the DDPG Agent. Having not worked with the algorith before, I decided to use TF Agents to implement the algorithm so I could get familiar with the set up and design. Furthermore, I wanted to get the DDPG Agent working on the OpenAI Gym Minataur scene first, to confirm that my set up will work.
This has been done, and can be found under CoppeliaSim\DDPG\TF Agent\_main_.py. The implementation is essentially the same as that in the paper, except that I allowed my network to run for an entire epsidoe, learn from it, then iterate again as oppose to having it done at each step. This is likely counterproductive but I wanted to do some experimentation with how the network learns. I am currently still adjusting the hyperparameters to improve the rate of convergence, but have had the following results:
https://docs.google.com/document/d/1ce6UHoxpXv4iciYGPrLfzhLmyC_gcSpX-gh4j7mdMfM/edit?usp=sharing (Link to Experimentation Log)

After I am satisfied with my parameter tuning, I will be recoding the DDPG algorithm from just tensorflow. This will give me much more flexibility in my implementation, allowing me to experiment with different types of algorithms. After that, I will code up the Coppelia Scene and train the DDPG Agent to pick up the box.

#### Known Bugs
DDPG None Error Bug:
For some reason when this runs, it throws a None value error occasionally. To fix, will need to add an if statement to just convert the None value to a 0. I haven't been able to workout what causes this bug, but have used the above temporary solution to sidestep the problem.

Coppelia Sim Gripper Bug:
Sending a torque to the gripper motor causes the fingers to move, but not to the point of closing. As far as I can tell, it's due to a problem with the child script on the P_Grip_Straight
