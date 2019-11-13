from __future__ import print_function, absolute_import, division
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import os
import nets.Network as Segception
import argparse
from utils.utils import get_params, restore_state, init_model, inference
import cv2
from collections import namedtuple

# enable eager mode
tf.enable_eager_execution()
tf.set_random_seed(7)
np.random.seed(7)

Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'eventId'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'color'       , # The color of this label
    ] )
labels = [
    #       name                     eventId    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  255 ,      255 , 'void'            , 0       ,  (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  255 ,      255 , 'void'            , 0       , (  0,  0,  0) ),
    Label(  'rectification border' ,  255 ,      255 , 'void'            , 0       , (  0,  0,  0) ),
    Label(  'out of roi'           ,  255 ,      255 , 'void'            , 0       , (  0,  0,  0) ),
    Label(  'static'               ,  255 ,      255 , 'void'            , 0       , (  0,  0,  0) ),
    Label(  'dynamic'              ,  255 ,      255 , 'void'            , 0       , (111, 74,  0) ),
    Label(  'ground'               ,  255 ,      255 , 'void'            , 0       , ( 81,  0, 81) ),
    Label(  'road'                 ,  0 ,        0 , 'flat'            , 1       , (128, 64,128) ),
    Label(  'sidewalk'             ,  0 ,        1 , 'flat'            , 1      ,  (244, 35,232) ),
    Label(  'parking'              ,  0 ,      255 , 'flat'            , 1       , (250,170,160) ),
    Label(  'rail track'           , 0 ,      255 , 'flat'            , 1       , (230,150,140) ),
    Label(  'building'             , 1 ,        2 , 'construction'    , 2       , ( 70, 70, 70) ),
    Label(  'wall'                 , 1 ,        3 , 'construction'    , 2       , (102,102,156) ),
    Label(  'fence'                , 1 ,        4 , 'construction'    , 2      ,  (190,153,153) ),
    Label(  'guard rail'           , 1 ,      255 , 'construction'    , 2       , (180,165,180) ),
    Label(  'bridge'               , 1 ,      255 , 'construction'    , 2       , (150,100,100) ),
    Label(  'tunnel'               , 1 ,      255 , 'construction'    , 2       , (150,120, 90) ),
    Label(  'pole'                 , 2 ,        5 , 'object'          , 3       , (153,153,153) ),
    Label(  'polegroup'            , 2 ,      255 , 'object'          , 3       , (153,153,153) ),
    Label(  'traffic light'        , 2 ,        6 , 'object'          , 3      ,  (250,170, 30) ),
    Label(  'traffic sign'         , 2 ,        7 , 'object'          , 3      ,  (220,220,  0) ),
    Label(  'vegetation'           , 3 ,        8 , 'nature'          , 4      ,  (107,142, 35) ),
    Label(  'terrain'              , 3 ,        9 , 'nature'          , 4     ,   (152,251,152) ),
    Label(  'sky'                  , 1 ,       10 , 'sky'             , 5      ,  ( 70,130,180) ),
    Label(  'person'               , 4 ,       11 , 'human'           , 6       ,  (220, 20, 60) ),
    Label(  'rider'                , 4 ,       12 , 'human'           , 6       ,  (255,  0,  0) ),
    Label(  'car'                  , 5 ,       13 , 'vehicle'         , 7       ,  (  0,  0,142) ),
    Label(  'truck'                , 5 ,       14 , 'vehicle'         , 7       ,  (  0,  0, 70) ),
    Label(  'bus'                  , 5 ,       15 , 'vehicle'         , 7       ,  (  0, 60,100) ),
    Label(  'caravan'              , 5 ,      255 , 'vehicle'         , 7       , (  0,  0, 90) ),
    Label(  'trailer'              , 5 ,      255 , 'vehicle'         , 7       , (  0,  0,110) ),
    Label(  'train'                , 5 ,       16 , 'vehicle'         , 7       ,  (  0, 80,100) ),
    Label(  'motorcycle'           , 5 ,       17 , 'vehicle'         , 7       ,  (  0,  0,230) ),
    Label(  'bicycle'              , 5 ,       18 , 'vehicle'         , 7       ,  (119, 11, 32) ),
    Label(  'license plate'        , 255 ,       -1 , 'vehicle'         , 7       , (  0,  0,142) ),
]

trainId2label   = { label.trainId : label for label in reversed(labels) }
def fromIdTrainToId(imgin):
    imgout = imgin.copy()
    for idTrain in trainId2label:
        imgout[imgin == idTrain] = trainId2label[idTrain].eventId
    return imgout

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", help="image path", default='/media/snowflake/Data/city/images/train/aachen_000003_000019_gtFine_labelIds.png')
    parser.add_argument("--model_path", help="Model path", default='weights/cityscapes_grayscale')
    parser.add_argument("--n_classes", help="number of classes to classify", default=19)
    parser.add_argument("--width", help="number of epochs to train", default=352)
    parser.add_argument("--height", help="number of epochs to train", default=224)
    parser.add_argument("--n_gpu", help="number of the gpu", default=0)
    args = parser.parse_args()

    # some parameters
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.n_gpu)
    n_classes = int(args.n_classes)
    width = int(args.width)
    height = int(args.height)
    channels = 1
    name_best_model = os.path.join(args.model_path, 'best')


    # build model and optimizer
    model = Segception.Segception_v4(num_classes=n_classes, weights=None, input_shape=(None, None, channels))

    # Init models (optional, just for get_params function)
    init_model(model, input_shape=(1, width, height, channels))

    variables_to_restore = model.variables
    variables_to_save = model.variables
    variables_to_optimize = model.variables

    # Init saver. can use also ckpt = tfe.Checkpoint((model=model, optimizer=optimizer,learning_rate=learning_rate, global_step=global_step)
    saver_model = tfe.Saver(var_list=variables_to_save)
    restore_model = tfe.Saver(var_list=variables_to_restore)

    # restore if model saved and show number of params
    restore_state(restore_model, name_best_model)
    get_params(model)


    img = cv2.imread(args.image_path, 0)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA).astype(np.float32)
    img = np.expand_dims(img, -1)
    img = np.expand_dims(img, 0)
    print(img.shape)

    prediction = inference(model, img, n_classes, flip_inference=True, scales=[0.75, 1, 1.5], preprocess_mode=None)
    print(prediction.numpy().shape)
    prediction = tf.argmax(prediction, -1)
    print(prediction.numpy().shape)

    img = np.squeeze(img).astype(np.uint8)
    prediction = np.squeeze(prediction.numpy()).astype(np.uint8)
    prediction_6classes = fromIdTrainToId(prediction).astype(np.uint8)

    cv2.imshow('image', img)
    cv2.imshow('pred (cityscapes classes)', prediction*13) # *13 for visualization
    cv2.imshow('pred (event classes)', prediction_6classes*40)# *40 for visualization
    cv2.waitKey(0)
    cv2.destroyAllWindows()
