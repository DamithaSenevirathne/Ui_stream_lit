#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Imu
from sensor_msgs.msg import Image
import cv2

from keras.models import load_model
from tensorflow.python.keras.backend import set_session #

import joblib
import numpy as np
import tensorflow as tf


n_features=3
look_back=3
error=[0]
scaler_filename = "/models_imu/scaler_for_linear_acceleration"
model_filename="/models_imu/LSTM_AutoEncoder_3_steps.hdf5"
global temp
scaler = joblib.load(scaler_filename) 

global model#
global graph#
global sess #
graph = tf.get_default_graph()
sess = tf.Session(config=tf.compat.v1.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=True))

model=load_model(model_filename)
temp=[]
def callback(data):
    #rospy.loginfo(rospy.get_caller_id() + " I heard "+ str(data.data))
    pass

def ImuCallback(IMUdata):
	#rospy.loginfo(rospy.get_caller_id() + " I heard imu %d", IMUdata.header.seq)
	L = [float(IMUdata.linear_acceleration.x),float(IMUdata.linear_acceleration.y),float(IMUdata.linear_acceleration.z)]
	L = np.array([L])

	threshold = 0.1


	#print(L.shape)
	i = 0
	is_True  = False
	count = 0

	L = scaler.transform(L)
	print(L,type(L[0]))
	if(count ==1):
		is_True = True
		
	while(count==0):
		temp.append(L) 
		
		if(i==2):
			count+=1
			
			temp=np.array(temp).reshape(1,look_back,n_features)
			#print(temp.shape)
			print(temp)
			with graph.as_default():#
				set_session(sess)#
				pred=model.predict(temp)
				error = np.mean(np.mean(np.abs(pred-temp),axis=1),axis=0)
			
			if(error[0]>threshold):
				print("ANOMALY")
			else:
				print("NORMAL")
	  
			break
		   
		i+=1

	if(is_True):
		
		temp =  np.append(temp[0][1:],L,axis = 0)
		temp = np.array([temp])
		temp=temp.reshape(1,look_back,n_features)
		
		with graph.as_default():#
			set_session(sess)#
			pred=model.predict(temp)  

			error = np.mean(np.mean(np.abs(pred-temp),axis=1),axis=0)
		    
		if(error[0]>threshold):
		    print("ANOMALY")
		else:
		    print("NORMAL")
		count+=1

	

def ImageCallback(Imagedata):
    #rospy.loginfo(Imagedata.header)
    #rospy.loginfo(Imagedata.height)
    #rospy.loginfo(Imagedata.width)
    #rospy.loginfo(Imagedata.encoding)
    #rospy.loginfo(Imagedata.is_bigendian)
    #rospy.loginfo(Imagedata.step)

    a = np.array(list(map(int,list(Imagedata.data))))
    #print(type(a[0]))
    b = np.reshape(a[::-1], (Imagedata.height,Imagedata.width))  

    
    #cv2.imwrite('my'+str(Imagedata.header.seq)+'.png',b)
def listener():

    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber("chatterbox", String, callback)

    rospy.Subscriber("mavros/imu/data", Imu, ImuCallback)

    rospy.Subscriber("/pylon_camera_node/image_raw", Image, ImageCallback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
