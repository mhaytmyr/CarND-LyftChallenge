
LABEL_MAP = {0:'None',1:'Buildings',2:'Fences',3:'Other',4:'Pedestrians',
        5:'Poles',6:'RoadLines',7:'Roads',8:'Sidewalks',9:'Vegetation',
        10:'Vehicles',11:'Walls',12:'TrafficSigns'};

COLOR_MAP = {0: [0,0,0],1: [0,1,0], 2: [1,0,0], 3:[ 1,1,0], 
        4: [1,0,0], 5: [0,0,1], 6: [192/255.,192/255.,192/255.], 7: [128/255.,128/255.,0], 
        8: [128/255.,0,0], 9: [ 0,128/255.,0], 10: [128/255.,128/255.,128/255.], 
        11: [128/255.,0,128/255.,], 12: [ 0,128/255.,128/255.]};

TOPCROP, BOTTOMCROP = 126, 526; #500;
H,W,C,numClasses = (BOTTOMCROP-TOPCROP),800,3,3 


########################
#### CONFIG ResNet #####
filter1 = 12;
filter2 = 24;
filter3 = 32;
filter4 = 64;
filter5 = 128;
filter6 = 256;
batchSize = 8;
numEpochs = 4;
H0,W0 = 256,256

modelName = "1x256x256_ResNet_1C1024_1C528_3U256_3U128_3U64_3U32_3U24_3U12"
########################

