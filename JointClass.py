# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 16:13:47 2020

@author: Percy
"""
import b0RemoteApi
class JointClass:
    def __init__(self,name):
        self.name = name
        
    def jointAngleCallback(msg):
        b0RemoteApi.RemoteApiClient('b0RemoteApi_pythonClient','b0RemoteApi').jointAngle=msg[1]
    def 