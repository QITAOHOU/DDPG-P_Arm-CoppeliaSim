# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 15:49:13 2020

@author: Percy
"""
from PIL import Image
from PIL import ImageColor
from io import BytesIO
import msgpack
import b0RemoteApi
import time
with b0RemoteApi.RemoteApiClient('b0RemoteApi_pythonClient','b0RemoteApi') as client:
    client.doNextStep=True
    client.runInSynchronousMode=True
    def simulationStepStarted(msg):
        simTime=msg[1][b'simulationTime'];
        print('Simulation step started. Simulation time: ',simTime)
        
    def simulationStepDone(msg):
        simTime=msg[1][b'simulationTime'];
        print('Simulation step done. Simulation time: ',simTime);
        client.doNextStep=True
    def Unpacker(msg):
        print("1")
        unpacked = msgpack.unpackb(msg)
        print("Unpacked!")
        out = Image.fromarray(unpacked, "RGB",)
        print("Image Created!")
        out.show()
        return out
    def imageCallback(msg):
        print('Received image.',msg[1])
        # print(msg[2])
        # print(type(msg[2]))
        ByteHexTo1DPythonList(msg)
        # print(int(msg[2],16))
        # I=BytesIO(msg[2])
        # I.getvalue()
        #ITS IN HEX Numbers!
        ImageColor.getcolor(msg[2],"RGB")

        # I = msgpack.unpackb(msg[2])
        # print(I)
#        client._handleReceivedMessage(self,msg)
        client.simxSetVisionSensorImage(passiveVisionSensorHandle[1],False,msg[2],client.simxCreatePublisher())
        #client.simxSetVisionSensorImage()
#    def ShowImage(msg):
#        print("It's happened again")

    def ByteHexTo1DPythonList(msg):
        # view = msg[2].getbuffer()
        # view[25:]
        # msg[2].decode("utf-16")
        NumVal = list(msg[2])
        #LIST OF Numerical Values for Picture
        print(len(NumVal))
        # DecodedHex = msg[2].decode("ascii")
        # print(DecodedHex)
    def stepSimulation():
        if client.runInSynchronousMode:
            while not client.doNextStep:
                client.simxSpinOnce()
            client.doNextStep=False
            client.simxSynchronousTrigger()
        else:
            client.simxSpinOnce()

    client.simxAddStatusbarMessage('Hello world!',client.simxDefaultPublisher())
    visionSensorHandle=client.simxGetObjectHandle('Vision_sensor',client.simxServiceCall())
    passiveVisionSensorHandle=client.simxGetObjectHandle('PassiveVisionSensor',client.simxServiceCall())

    if client.runInSynchronousMode:
        client.simxSynchronous(True)
    dedicatedSub=client.simxCreateSubscriber(imageCallback,1,True)
#    dedicatedSub2=client.simxCreateSubscriber(ShowImage,1,True)
    client.simxGetVisionSensorImage(visionSensorHandle[1],False,dedicatedSub)
#    msgpack.unpack()
    #client.simxGetVisionSensorImage(visionSensorHandle[1],False,client.simxCreateSubscriber(imageCallback))
    
    client.simxGetSimulationStepStarted(client.simxDefaultSubscriber(simulationStepStarted));
    client.simxGetSimulationStepDone(client.simxDefaultSubscriber(simulationStepDone));
    client.simxStartSimulation(client.simxDefaultPublisher())
    startTime=time.time()
    while time.time()<startTime+50:
        stepSimulation()
    
    print("Panic!")  
    client.simxStopSimulation(client.simxDefaultPublisher())

    