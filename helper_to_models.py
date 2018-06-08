import keras as K
from keras.engine import Layer
from keras.engine import InputSpec
from keras.utils.np_utils import to_categorical
from keras.utils import conv_utils
from keras.engine.topology import get_source_inputs
from keras.applications import imagenet_utils
from keras.utils import conv_utils

from keras.backend import tf as ktf
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import SeparableConv2D, MaxPooling2D, UpSampling2D, Conv2D
from keras.applications.mobilenet import DepthwiseConv2D
from keras.layers import Lambda, Input, BatchNormalization, Concatenate, ZeroPadding2D, Add
from keras.models import Model, Sequential, model_from_json
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.applications import ResNet50

from config import *
import numpy as np, json

class BilinearUpsampling(Layer):
    def __init__(self, upsampling=(2, 2), output_size=None, data_format=None, **kwargs):
        super(BilinearUpsampling, self).__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=4)
        if output_size:
            self.output_size = conv_utils.normalize_tuple(output_size, 2, 'output_size')
            self.upsampling = None
        else:
            self.output_size = None
            self.upsampling = conv_utils.normalize_tuple(upsampling, 2, 'upsampling')

    def compute_output_shape(self, input_shape):
        if self.upsampling:
            height = self.upsampling[0] * input_shape[1] if input_shape[1] is not None else None
            width = self.upsampling[1] * input_shape[2] if input_shape[2] is not None else None
        else:
            height = self.output_size[0]
            width = self.output_size[1]
        
        return (input_shape[0],height,width,input_shape[3])
    
    def call(self, inputs):
        if self.upsampling:
            return ktf.image.resize_bilinear(inputs, 
                                (inputs.shape[1] * self.upsampling[0],inputs.shape[2] * self.upsampling[1]),
                                align_corners=True)
        else:
            return ktf.image.resize_bilinear(inputs, 
                                (self.output_size[0],self.output_size[1]),
                                align_corners=True)
            
    def get_config(self):
        config = {'upsampling': self.upsampling,
                'output_size': self.output_size,
                'data_format': self.data_format}

        base_config = super(BilinearUpsampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def relu6(x):
    return K.activations.relu(x, max_value=6)

#testing model
def resnet_2048():
    upKernel = (3,3)
    #main input image
    input_ = Input(shape=(H,W,C));
    #resize image here
    data = BilinearUpsampling(output_size=(H0,W0))(input_)

    #create base model
    baseModel = ResNet50(weights='imagenet',include_top=False,pooling=None,input_tensor=data);

    #get down layers
    baseInput = baseModel.get_layer('input_1').output;
    print('Base In',baseInput.shape)
    baseOut = baseModel.get_layer('activation_49').output; 
    print('Base Out',baseOut.shape)
    down4_res = baseModel.get_layer('activation_40').output;
    print('Down4',down4_res.shape)
    down3_res = baseModel.get_layer('activation_22').output;
    print('Down3',down3_res.shape)
    down2_res = baseModel.get_layer('activation_10').output;
    print('Down2',down2_res.shape)
    down1_res = baseModel.get_layer('activation_1').output;
    print('Down1',down1_res.shape)

    center = Conv2D(1024, (1, 1), padding='same',kernel_initializer='glorot_uniform')(baseOut)
    center = BatchNormalization(epsilon=1e-4)(center)
    center = LeakyReLU(alpha=0.03)(center) 
    center = Conv2D(528, (1, 1), padding='same',kernel_initializer='glorot_uniform')(center)
    center = BatchNormalization(epsilon=1e-4)(center)
    center = LeakyReLU(alpha=0.03)(center) 
    print('Center ',center.shape)

    up5 = resUp(filter6, upKernel, center,down4_res)
    print('Up5 ',up5.shape)
    up4 = resUp(filter5, upKernel, up5, down3_res)
    print('Up4 ',up4.shape) 
    up3 = resUp(filter4, upKernel,up4, down2_res)
    print('Up3 ',up3.shape) 
    up2 = resUp(filter3, (3,3), up3, down1_res)
    print('Up2 ',up2.shape) 
    up1 = pyramidUp(filter2, upKernel, up2, baseInput,(H//2,W//2))
    print('Up1 ',up1.shape)
    up0 = pyramidUp(filter1, upKernel, up1, input_,(H,W))
    print('Up0 ',up0.shape)

    #create softmax later
    x = Conv2D(numClasses,(1,1),strides=(1,1),activation=relu6,kernel_initializer='glorot_uniform')(up0);
    classify = Activation('softmax')(x);
    
    #create model
    model = Model(inputs=input_, outputs=classify)
    
    #set basemodel parameters to non-trainable
    for layer in baseModel.layers:
        layer.trainable = False
    
    return model

def pyramidUp(filters, upKernel, input_, down_, shape_):
    #upsample input layer
    up_ = BilinearUpsampling(output_size=shape_)(input_)
    #upsample residual layer
    down_ = BilinearUpsampling(output_size=shape_)(down_)

    up_ = Conv2D(filters, upKernel, padding='same', kernel_initializer='glorot_uniform')(up_)
    up_ = BatchNormalization(epsilon=1e-4,axis=-1)(up_)
    up_ = LeakyReLU(alpha=0.03)(up_) 
    up_ = Concatenate(axis=-1)([down_, up_])
    up_ = Conv2D(filters, upKernel, padding='same', kernel_initializer='glorot_uniform')(up_)
    up_ = BatchNormalization(epsilon=1e-4,axis=-1)(up_)
    up_ = LeakyReLU(alpha=0.03)(up_) 
    up_ = Conv2D(filters, upKernel, padding='same', kernel_initializer='glorot_uniform')(up_)
    up_ = BatchNormalization(epsilon=1e-4,axis=-1)(up_)
    up_ = LeakyReLU(alpha=0.03)(up_) 
    return up_

def resUp(filters, upKernel, input_, down_):
   
    inputShape = (down_.shape[1].value,down_.shape[2].value)
    up_ = BilinearUpsampling(output_size=inputShape)(input_)
    up_ = Conv2D(filters, upKernel, padding='same', kernel_initializer='glorot_uniform')(up_)
    up_ = BatchNormalization(epsilon=1e-4,axis=-1)(up_)
    up_ = LeakyReLU(alpha=0.03)(up_) 
    up_ = Concatenate(axis=-1)([down_, up_])
    up_ = Conv2D(filters, upKernel, padding='same', kernel_initializer='glorot_uniform')(up_)
    up_ = BatchNormalization(epsilon=1e-4,axis=-1)(up_)
    up_ = LeakyReLU(alpha=0.03)(up_) 
    up_ = Conv2D(filters, upKernel, padding='same', kernel_initializer='glorot_uniform')(up_)
    up_ = BatchNormalization(epsilon=1e-4,axis=-1)(up_)
    up_ = LeakyReLU(alpha=0.03)(up_) 
    return up_


