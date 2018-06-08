import numpy as np
import json
from helper_to_models import *
from config import *

import tensorflow as tf
import keras as K
from keras.callbacks import Callback
from keras.models import load_model, model_from_json
from keras.utils.np_utils import to_categorical
from keras.losses import categorical_crossentropy


def precision_recall(y_true, y_pred,smooth=1):
    
    true_positives = K.backend.sum(y_true * y_pred,(1,2)) #find TP per image (batchSize,chanels)
    possible_positives = K.backend.sum(y_true,(1,2))+smooth 
    predicted_positives = K.backend.sum(y_pred,(1,2))+smooth

    recall = true_positives / (possible_positives)
    precision = true_positives / (predicted_positives)

    return K.backend.mean(recall), K.backend.mean(precision) #then take average over batchSize

def fbeta_score(y_true, y_pred, beta=1, smooth=1):
    if beta < 0: raise ValueError('The lowest choosable beta is zero (only precision).')
    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.backend.sum(K.backend.round(K.backend.clip(y_true, 0, 1))) == 0: return 0
    
    r,p = precision_recall(y_true,y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.backend.epsilon())
    return fbeta_score

def fbeta_vehicle(y_true,y_pred,smooth=1):
    
    #first convert y_pred to categorical
    y_true = K.backend.cast(K.backend.equal(K.backend.argmax(y_true,axis=-1),2),'float32');
    y_pred = K.backend.cast(K.backend.equal(K.backend.argmax(y_pred,axis=-1),2),'float32');
    
    return fbeta_score(y_true,y_pred,2)

def fbeta_road(y_true,y_pred):
    
    #first convert y_pred to categorical
    y_true = K.backend.cast(K.backend.equal(K.backend.argmax(y_true,axis=-1),1),'float32');
    y_pred = K.backend.cast(K.backend.equal(K.backend.argmax(y_pred,axis=-1),1),'float32');

    return fbeta_score(y_true,y_pred,0.5)

# Ref: salehi17, "Twersky loss function for image segmentation using 3D FCDN"
# -> the score is computed for each class separately and then summed
# alpha=beta=0.5 : dice coefficient
# alpha=beta=1   : tanimoto coefficient (also known as jaccard)
# alpha+beta=1   : produces set of F*-scores, larger beta places emphasis on false positivies
def tversky_loss(y_true, y_pred):
    alpha = 0.3
    beta  = 0.7
                
    p0 = y_pred     # proba that voxels are class i
    p1 = 1-y_pred   # proba that voxels are not class i
    g0 = y_true
    g1 = 1-y_true
                                        
    num = K.backend.sum(p0*g0, axis=(0,1,2)) #compute loss per-batch
    den = num + alpha*K.backend.sum(p0*g1,axis=(0,1,2)) + beta*K.backend.sum(p1*g0,axis=(0,1,2))
    T = K.backend.sum(num/den)

    return numClasses-T

def tot_loss(y_true, y_pred):
    return categorical_crossentropy(y_true, y_pred) + tversky_loss(y_true, y_pred)

def load_json_model(modelName):
    filePath = './checkpoint/'+modelName+".json";
    fileWeight = './checkpoint/'+modelName+"_weights.h5"
   
    with open(filePath,'r') as fp:
        json_data = fp.read();
    model = model_from_json(json_data,custom_objects={'relu6':relu6,'BilinearUpsampling':BilinearUpsampling})
    model.load_weights(fileWeight)

    return model

def load_frozen_model(modelName):
    frozenGraphPath = './checkpoint/'+modelName+".pb";
    # We load the protobuf file from the disk and parse it to retrieve the  unserialized graph_def
    with tf.gfile.GFile(frozenGraphPath, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        # Then, we can use again a convenient built-in function to import a graph_def into the 
        # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, 
                            input_map=None, 
                            return_elements=None, 
                            name="prefix", 
                            op_dict=None, 
                            producer_op_list=None
                            )

    return graph


def train_model(trainGen,valGen,stepsPerEpoch,numEpochs,valSteps):
    try:
        model = load_json_model(modelName)
        print("Loading model...");
    except Exception as e:
        print(e);
        print("Creating new model...")
        model = resnet_2048();

    print(model.summary())
    
    #save only model
    with open('./checkpoint/'+modelName+'.json','w') as fp:
        fp.write(model.to_json());

    rmsprop = K.optimizers.RMSprop(
            lr=0.0001, #global learning rate,  
            rho=0.95, #exponential moving average; r = rho*initial_accumilation+(1-rho)*current_gradient  
            epsilon=1e-6, #small constant to stabilize division by zero
            decay=0.0
            )
    #compile model
    model.compile(optimizer=rmsprop,
                    loss = tot_loss,
                    #metrics=["categorical_accuracy"]
                    metrics=[fbeta_vehicle,fbeta_road]
                    );

    #define callbacks
    modelCheckpoint = K.callbacks.ModelCheckpoint("./checkpoint/"+modelName+"_weights.h5",
                                'val_loss',
                                verbose=0, 
                                save_best_only=True, 
                                save_weights_only=True, 
                                mode='min', period=1)

    reduceLearningRate = K.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                factor=0.1, patience=3, 
                                verbose=0, mode='auto', 
                                cooldown=1, min_lr=0)

    earlyStopping = K.callbacks.EarlyStopping(monitor='val_loss', 
                                patience=3, 
                                verbose=1, 
                                min_delta = 0.0001,                                                                                                            mode='min')
    validationMetric = Metrics(valGen,valSteps,batchSize);

    #fit model and store history
    hist = model.fit_generator(trainGen,
              steps_per_epoch = stepsPerEpoch,
              epochs=numEpochs,
              validation_data = valGen,
              validation_steps = valSteps,
              verbose=1,
              callbacks=[modelCheckpoint,reduceLearningRate])

        
