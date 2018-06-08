import numpy as np
import cv2
from config import *
from keras.utils.np_utils import to_categorical

def pre_process_img(img,colorChannel='RGB'):
    #first remove the hood of car
    img = img[TOPCROP:BOTTOMCROP,...];

    #normalize image
    return img/255.-0.5

def pre_process_label(label):
    #only use third index
    label = label[:,:,2];
    
    #set car hood to zero
    mask = (label[-115:,:]==10)
    mask = np.pad(mask,((485,0),(0,0)),mode='constant');
    label[mask] = 0;
    
    label = label[TOPCROP:BOTTOMCROP,:];

    #set pixels labeled as lane markings(value=6) to be same as the road surface(value=7)
    mask = (label==6)
    label[mask] = 7;

    #now set anything that is not vehicle or drivable surface to 'Other'
    mask = (label==7) + (label==10);
    label[~mask] = 0;

    #now set vehicles and roads
    label[(label==7)] = 1; #roads
    label[(label==10)] = 2; #vehicles

    return label

def augment_data(img,label):
    choice = np.random.choice(['flip','flip','shift','noise','rotate']);

    if choice=='rotate':
        M_rot = cv2.getRotationMatrix2D((W/2,H/2),np.random.randint(-15,15),1);
        img = cv2.warpAffine(img,M_rot,(W,H));
        label = cv2.warpAffine(label,M_rot,(W,H));
    elif choice=='shift':
        M_shift = np.float32([[1,0,np.random.randint(0,50)],[0,1,np.random.randint(0,50)]])
        img = cv2.warpAffine(img,M_shift,(W,H))
        label = cv2.warpAffine(label,M_shift,(W,H));
    elif choice=='noise':
        mean = 0.0   # some constant
        std = 0.1    # some constant (standard deviation)
        img = img + np.random.normal(mean, std, img.shape)
    elif choice=='zoom':
        zoomfactor = np.random.randint(1,8)
        M_zoom = cv2.getRotationMatrix2D((W/2,H/2), 0, zoomfactor) 
        img = cv2.warpAffine(img, M_zoom,(W,H))
        label = cv2.warpAffine(label, M_zoom,(W,H))
    elif choice=='flip':
        img = cv2.flip(img,1);
        label = cv2.flip(label,1);

    return img,label
    
def data_generator(imgs,labels,batchSize=50,augment=True):
    
    #initialize pointer
    idx,n = 0,len(imgs);

    #input data 
    X_in = np.zeros((batchSize,H,W,C),dtype='float32');
    X_out = np.zeros((batchSize,H,W,numClasses),dtype='float32');
    
    #yield data with or w/o augmentation    
    while True:
        for i in range(batchSize):
            img = cv2.imread(imgs[idx%n]);
            img = pre_process_img(img);

            #get label data
            label = cv2.imread(labels[idx%n]);
            label = pre_process_label(label);
            
            if augment:
                img, label = augment_data(img,label)
            
            X_in[i,:,:,:] = img.astype('float32');
            X_out[i,:,:,:] = to_categorical(label,num_classes=numClasses).reshape((H,W,numClasses)).astype('float32');

            #increment counter
            idx+=1;

        yield X_in, X_out


