import sys, skvideo.io, json
from skvideo.io import FFmpegWriter
import numpy as np
import cv2
import tensorflow as tf
from config import *
from helper_to_train import load_frozen_model

class MyPrediction(object):
    def __init__(self,modelPath):  
        self.graph = load_frozen_model(modelPath)
    
    def pre_process_img(self,inputImg):
        #my files are trained on BGR, convert same format
        img = inputImg[...,::-1];
        img = img[:,TOPCROP:BOTTOMCROP,...];
        return img/255.-0.5
    
    def post_process_img(self,img):
        img = np.pad(img,((0,0),(TOPCROP,600-BOTTOMCROP),(0,0)),mode='constant',constant_values=0);
        binary_car_result = np.where(img==2,1,0).astype('uint8')
        binary_road_result = np.where(img==1,1,0).astype('uint8')
        
        return binary_car_result, binary_road_result
    
    def predict(self,imgs,sess):
        inputName = self.graph.get_operations()[0].name;
        outputName = self.graph.get_operations()[-1].name;

        tf_X = self.graph.get_tensor_by_name(inputName+':0')
        tf_Y = self.graph.get_tensor_by_name(outputName+':0')
        car_results, road_results = np.zeros(imgs.shape[:-1]), np.zeros(imgs.shape[:-1])
        
        X = self.pre_process_img(imgs[np.newaxis,...])
        pred = np.argmax(sess.run(tf_Y,feed_dict={tf_X:X}),axis=3)
        img = np.pad(pred,((0,0),(TOPCROP,600-BOTTOMCROP),(0,0)),mode='constant',constant_values=0);   
        return img


fileName = sys.argv[-1]

if fileName == 'predict_video.py':
    print ("Error loading video")
    quit

#read in video    
videoIn = skvideo.io.vread(fileName)
idx, N = 0, videoIn.shape[0]

videoOut = FFmpegWriter(fileName.replace('.mp4','_pred.mp4'))

# Initialize prediction class
myPredictor = MyPrediction(modelName);

#print(binary_car_results.shape, binary_road_results.shape)
COLOR_MAP = {0: [0,0,0],1: [0,255,0], 2: [255,0,0]}

# We launch a session
with tf.Session(graph=myPredictor.graph) as sess:    
            
    while True:
        frame = videoIn[idx%N,...,::-1] 
        Y = myPredictor.predict(videoIn[idx%N,...],sess);

        pred = np.zeros_like(frame);
        for item in range(3):
            mask = (Y[0,...]==item)
            pred[mask] = COLOR_MAP[item]

        #create blended object
        framePred = cv2.addWeighted(frame,0.6,pred,0.4,10)
        videoOut.writeFrame(framePred); idx+=1; 
        print("Processing frame {}".format(idx))

    cv2.destroyAllWindows();
