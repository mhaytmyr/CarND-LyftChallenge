import glob
import numpy as np
import pickle, sys
from sklearn.model_selection import train_test_split

from helper_to_visualize import *
from helper_prepare_data import *
from helper_to_train import *
from config import *

def plot_generator(myGen):
    
    while True:
        X,Y = next(myGen);
        print(X.shape, Y.shape)
        label = np.zeros(X.shape[1:]);

        for item in range(3):
            mask = (Y[0,:,:,item]==1)
            print(mask.shape,label.shape)
            label[mask] = COLOR_MAP[item];

        img = X[0,...]+0.5;
        cv2.imshow("Test",np.hstack([img,label]));
        k=cv2.waitKey(0);

        if k==27:
            break

def plot_prediction(valGen,model):
    while True:
        X,Y = next(valGen)

        #label
        truth = np.zeros_like(X[0,:,:,:])
        pred_img = np.zeros_like(truth)
        pred = np.argmax(model.predict(X),axis=3)
        
        for item in range(3):
            mask = (Y[0,:,:,item]==1)
            truth[mask] = COLOR_MAP[item];
            mask = (pred[0,...]==item)
            pred_img[mask] = COLOR_MAP[item];

        cv2.imshow('Test',np.hstack([pred_img,truth]))
        k = cv2.waitKey(0)

        if k==27:
            break

def plot_graph_prediction(valGen,graph):
    
    inputName = graph.get_operations()[0].name+':0';
    outName = graph.get_operations()[-1].name+':0';
    
    tf_X = graph.get_tensor_by_name(inputName)
    tf_Y = graph.get_tensor_by_name(outName)

    # We launch a session
    with tf.Session(graph=graph) as sess:
        while True:
            X,Y = next(valGen)

            #label
            truth = np.zeros_like(X[0,:,:,:])
            pred_img = np.zeros_like(truth)
            pred = np.argmax(sess.run(tf_Y,feed_dict={tf_X:X}),axis=3)
        
            for item in range(3):
                mask = (Y[0,:,:,item]==1)
                truth[mask] = COLOR_MAP[item];
                mask = (pred[0,...]==item)
                pred_img[mask] = COLOR_MAP[item];

            cv2.imshow('Test',np.hstack([pred_img,truth]))
            k = cv2.waitKey(0)

            if k==27:
                break

def convert_keras_tf(quantize=False):
    from keras import backend as K
    from tensorflow.python.framework import graph_util
    from tensorflow.python.framework import graph_io
    from tensorflow.python.tools import optimize_for_inference_lib
    
    K.set_learning_phase(0);
    K.set_image_data_format('channels_last');

    #firs load model
    net_model = load_json_model(modelName)

    #output name 
    output_model_file = modelName+'.pb';
    output_fld = './checkpoint/'

    #pred node names
    pred_node_names = ['output_node0']
    tf.identity(net_model.outputs[0], name=pred_node_names[0]);

    sess = K.get_session()
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)

    graph_io.write_graph(constant_graph, output_fld, output_model_file, as_text=False)
    print('saved the freezed graph (ready for inference) at: {}/{}'.format(output_fld,output_model_file))


if __name__=='__main__':

    trainData = glob.glob('./Train/CameraRGB/*')+glob.glob('./Test/*/CameraRGB/0001*');
    trainLabels = glob.glob('./Train/CameraSeg/*')+glob.glob('./Test/*/CameraSeg/0001*');
    #TODO, for now use train without order
    valData = glob.glob('./Val_2/CameraRGB/*');
    valLabels = glob.glob('./Val_2/CameraSeg/*');
    
    #randomly split train test samples
    X_train, X_test, y_train, y_test = train_test_split(trainData+valData,trainLabels+valLabels,test_size=0.3);
    trainGen = data_generator(X_train,y_train,batchSize=batchSize,augment=True)
    valGen = data_generator(X_test,y_test,batchSize=batchSize,augment=False)

    arg = sys.argv[1]
    if arg=='train':
        #train model
        train_model(trainGen,valGen,int(len(trainData)/batchSize),numEpochs,int(len(testData)/(batchSize)));
        #freeze model
        convert_keras_tf(quantize=False)
    elif arg=='test':
        #load frozen model
        graph = load_frozen_model(modelName)
        plot_graph_prediction(valGen,graph)
    
    elif arg=='plot':
        plot_generator(valGen)
