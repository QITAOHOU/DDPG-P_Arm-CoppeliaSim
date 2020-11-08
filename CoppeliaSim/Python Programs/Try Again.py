# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 15:49:13 2020

@author: Percy
"""
#from JointClass import JointClass
import pdb
import math
from PIL import Image
from PIL import ImageColor
from io import BytesIO
import b0RemoteApi
import time
with b0RemoteApi.RemoteApiClient('b0RemoteApi_pythonClient','b0RemoteApi') as client:
    client.doNextStep=True
    client.runInSynchronousMode=True
    client.jointAngle1=0
    client.jointAngle2=0   
    client.jointAngle3=0
    client.jointAngle4=0
    client.jointAngle5=0
    client.jointAngle6=0
    client.targetAngle=0
    client.maxForce=100
    jointAngleDict = {}
    def simulationStepStarted(msg):
        simTime=msg[1][b'simulationTime'];
        print('Simulation step started. Simulation time: ',simTime)  
    def simulationStepDone(msg):
        simTime=msg[1][b'simulationTime'];
        print('Simulation step done. Simulation time: ',simTime);
        client.doNextStep=True
    def imageCallback(msg):
        print('Received image.',msg[1])
        ByteHexTo1DPythonList(msg)
        client.simxSetVisionSensorImage(passiveVisionSensorHandle[1],False,msg[2],client.simxCreatePublisher())        
    def ByteHexTo1DPythonList(msg):
        NumVal = list(msg[2])
    def jointAngleCallback1(msg):
        client.jointAngle1=msg[1]
        jointAngleDict[str(pArmJointHandleList[0])] = client.jointAngle1
    def jointAngleCallback2(msg):
        client.jointAngle2=msg[1]
        jointAngleDict[str(pArmJointHandleList[1])] = client.jointAngle2
    def jointAngleCallback3(msg):
        client.jointAngle3=msg[1]
        jointAngleDict[str(pArmJointHandleList[2])] = client.jointAngle3
    def jointAngleCallback4(msg):
        client.jointAngle4=msg[1]
        jointAngleDict[str(pArmJointHandleList[3])] = client.jointAngle4
    def jointAngleCallback5(msg):
        client.jointAngle5=msg[1]
        jointAngleDict[str(pArmJointHandleList[4])] = client.jointAngle5
    def jointAngleCallback6(msg):
        client.jointAngle6=msg[1]
        jointAngleDict[str(pArmJointHandleList[5])] = client.jointAngle6
    def moveToAngle(jointH,angle):
        client.targetAngle=angle
        client.jointAngle = jointAngleDict[str(jointH)] 
        while abs(client.jointAngle-client.targetAngle)>0.1*math.pi/180:
            if client.doNextStep:
                client.doNextStep=False
                vel=computeTargetVelocity()
                client.simxSetJointTargetVelocity(jointH,vel,client.simxDefaultPublisher())
                client.simxSetJointMaxForce(jointH,client.maxForce,client.simxDefaultPublisher())
                client.simxSynchronousTrigger()
            client.simxSpinOnce()
            client.jointAngle = jointAngleDict[str(jointH)]    
        client.simxSetJointTargetVelocity(jointH,0,client.simxDefaultPublisher())
    def moveGripperMotor(targetVelocity,maxGripperForce):
        finger1 = client.simxGetObjectHandle("P_Grip_right_angle_finger1",client.simxServiceCall())[1]
        finger2= client.simxGetObjectHandle("P_Grip_right_angle_finger2",client.simxServiceCall())[1]
        collisionConditionFingerFinger = client.simxReadCollision(fingerFingerCollisionHandle,client.simxServiceCall())[1]
        collisionConditionFinger1Cuboid = client.simxReadCollision(fingerCuboidCollisionHandle,client.simxServiceCall())[1]
        collisionConditionFinger2Cuboid = client.simxReadCollision(fingerCuboidCollision2Handle,client.simxServiceCall())[1]

        client.simxSetJointTargetVelocity(pArmJointHandleList[6],targetVelocity,client.simxServiceCall())
        client.simxSetJointMaxForce(pArmJointHandleList[6],maxGripperForce,client.simxServiceCall())
        while collisionConditionFingerFinger == 0 or collisionConditionFinger2Cuboid == 0 or collisionConditionFinger1Cuboid==0:
            print("Hey!")
            client.simxSynchronousTrigger()
            client.simxSpinOnce()
            collisionConditionFingerFinger = client.simxReadCollision(fingerFingerCollisionHandle,client.simxServiceCall())[1]
            collisionConditionFinger1Cuboid = client.simxReadCollision(fingerCuboidCollisionHandle,client.simxServiceCall())[1]
            collisionConditionFinger2Cuboid = client.simxReadCollision(fingerCuboidCollision2Handle,client.simxServiceCall())[1]
    def computeTargetVelocity():
        dynStepSize=0.1
        velUpperLimit=360*math.pi/180
        PID_P=0.1
        errorValue=(client.targetAngle-client.jointAngle)
        sinAngle=math.sin(errorValue)
        cosAngle=math.cos(errorValue)
        errorValue=math.atan2(sinAngle,cosAngle)
        ctrl=errorValue*PID_P
        
        # Calculate the velocity needed to reach the position in one dynamic time step:
        velocity=ctrl/dynStepSize
        if (velocity>velUpperLimit):
            velocity=velUpperLimit
            
        if (velocity<-velUpperLimit):
            velocity=-velUpperLimit
        
        return velocity
    def stepSimulation():
        if client.runInSynchronousMode:
            while not client.doNextStep:
                client.simxSpinOnce()
            client.doNextStep=False
            client.simxSynchronousTrigger()
        else:
            client.simxSpinOnce()
    visionSensorHandle=client.simxGetObjectHandle('Vision_sensor',client.simxServiceCall())
    passiveVisionSensorHandle=client.simxGetObjectHandle('PassiveVisionSensor',client.simxServiceCall())
    pioneerHandle = client.simxGetObjectHandle("Pioneer_p3dx",client.simxServiceCall())
    parmHandle = client.simxGetObjectHandle("P_Arm",client.simxServiceCall())
    
    #List of Joint Handles, with each subsequent joint handle and ending with the P_Grip_Motor and Parent P_Arm Handle
    pArmJointHandleList = [client.simxGetObjectHandle("P_Arm_joint1",client.simxServiceCall())[1],client.simxGetObjectHandle("P_Arm_joint2",client.simxServiceCall())[1],client.simxGetObjectHandle("P_Arm_joint3",client.simxServiceCall())[1],client.simxGetObjectHandle("P_Arm_joint4",client.simxServiceCall())[1],client.simxGetObjectHandle("P_Arm_joint5",client.simxServiceCall())[1],client.simxGetObjectHandle("P_Arm_joint6",client.simxServiceCall())[1],client.simxGetObjectHandle("P_Grip_right_angle_motor",client.simxServiceCall())[1],client.simxGetObjectHandle("P_Arm",client.simxServiceCall())[1]]
    #List of JointAngleCallback functions
    jointAngleCallback = [jointAngleCallback1,jointAngleCallback2,jointAngleCallback3,jointAngleCallback4,jointAngleCallback5,jointAngleCallback6]
    #Filling Dictionary Entries
    jointAngleDict[str(pArmJointHandleList[0])] = 0
    jointAngleDict[str(pArmJointHandleList[1])] = 0
    jointAngleDict[str(pArmJointHandleList[2])] = 0
    jointAngleDict[str(pArmJointHandleList[3])] = 0
    jointAngleDict[str(pArmJointHandleList[4])] = 0
    jointAngleDict[str(pArmJointHandleList[5])] = 0
    #CollisionHandleCollection
    fingerFingerCollisionHandle=client.simxGetCollisionHandle("FingerFingerCollision",client.simxServiceCall())[1]
    fingerCuboidCollisionHandle=client.simxGetCollisionHandle("FingerCuboidCollision",client.simxServiceCall())[1]
    fingerCuboidCollision2Handle=client.simxGetCollisionHandle("FingerCuboidCollision2",client.simxServiceCall())[1]
    #Not really Sure What this Bit is For
    for i in range(len(pArmJointHandleList)-2):
        client.simxSetJointTargetVelocity(pArmJointHandleList[i],360*math.pi/180,client.simxServiceCall())
        client.simxGetJointPosition(pArmJointHandleList[i],client.simxDefaultSubscriber(jointAngleCallback[i]))
    client.simxSynchronous(True)
#    #ImageCollector    
#    dedicatedSub=client.simxCreateSubscriber(imageCallback,1,True)
#    client.simxGetVisionSensorImage(visionSensorHandle[1],False,dedicatedSub)
#    #Movement Executor
#    #Angle_List is Generated by NN

    
    #Creates the Relevant Subscriber/Publisher Channels for Stepping
    client.simxGetSimulationStepStarted(client.simxDefaultSubscriber(simulationStepStarted));
    client.simxGetSimulationStepDone(client.simxDefaultSubscriber(simulationStepDone));
    client.simxStartSimulation(client.simxDefaultPublisher())
    #Antiquated Time Based Stepping System
#    startTime=time.time()
#    while time.time()<startTime+5:
#        stepSimulation()
#    pdb.set_trace()
    #Sets Target Velocities to 0 to Lock Motors
    for i in range(len(pArmJointHandleList)-1):
        client.simxSetJointTargetVelocity(pArmJointHandleList[i],0,client.simxServiceCall())
    Angle_List = [0.3,-0.3,-0.3,0.3,0.3,1]
    #Executes PID Movement Method
#    for i in range(len(Angle_List)):
#        moveToAngle(pArmJointHandleList[i],Angle_List[i])
    #Moves Motor
    moveGripperMotor(1,1)
    client.simxStopSimulation(client.simxDefaultPublisher())

    