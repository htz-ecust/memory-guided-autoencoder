import imp
import os 
from yacs.config import CfgNode as CN

_C = CN()

_C.MODEL = CN()
_C.MODEL.NAME = 'MCA'

#-------------TRAING-------------------#
_C.TRAIN = CN()
_C.TRAIN.batch_size = 64
_C.TRAIN.lr = 0.1
_C.TRAIN.som_epoch = 200
_C.TRAIN.mae_epoch = 200
_C.TRAIN.device = 'gpu'
_C.TRAIN.pre_train_epoch = 0


#-------------DATA-------------------#
_C.DATA = CN()
_C.DATA.path = 'data/classData.csv'
_C.DATA.att_path = 'attribute_matrix.xls'
_C.DATA.test_index = [3,6,9]
_C.DATA.scale = True
_C.DATA.SPCA = True
_C.DATA.orth = False
_C.DATA.datatype = 'float32'


#---------------SOM---------------------#
_C.SOM = CN()
_C.SOM.som_training = 'self'
_C.SOM.weight_path = 'path'
_C.SOM.input_size = 400
_C.SOM.row = 20
_C.SOM.col = 20
_C.SOM.lr = 0.3
_C.SOM.sigma = 1.0
_C.SOM.neighborhood_function = 'mexican_hat'


#--------------ENCODER------------------#
_C.ENCODER = CN()
_C.ENCODER.num_node = [400,52]


#--------------DECODER------------------#
_C.DECODER = CN()
_C.DECODER.num_node = [52,400]

#--------------MAE------------------#
_C.MAE = CN()
_C.MAE.shrink = True
_C.MAE.shrink_thres = 0.005
_C.MAE.lamd_off = 0.005
_C.MAE.lamd_c = 1.0
_C.MAE.re_loss = 'cosine'
_C.MAE.num_me = 100
_C.MAE.similarity = 'cos'

#--------------EVALUATION------------------#
_C.EVALUATION = CN()
_C.EVALUATION.mode = 'zsl'
_C.EVALUATION.classifier = 'NB'
_C.EVALUATION.concate = 'x'
_C.EVALUATION.seen_clf = 'supervised'
_C.EVALUATION.metric = 'prc'


