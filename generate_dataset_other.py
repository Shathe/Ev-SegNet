import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import nets.Network as Segception
from utils.utils import preprocess, lr_decay, convert_to_tensors, restore_state, init_model, get_metrics, inference
import argparse
import cv2
import h5py
import glob
import os.path

tf.enable_eager_execution()
tf.set_random_seed(7)
np.random.seed(7)


def check_timestamps(on, off):
    max_value = max(max( on.flatten()) , max(off.flatten()))
    min_value = min(min( on.flatten()) ,min(off.flatten()))
    if min_value < -1 or max_value > 1:
        return True
    return False

def transform(image, class_to_category):
    return np.array([[class_to_category[str(y)] for y in x] for x in image])

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="Dataset to extract", default='/media/msrobot/discoGordo/Event-based/INIGO')
parser.add_argument("--weights", help="pretrained model", default='weights/best')
args = parser.parse_args()

n_gpu = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(n_gpu)
n_classes = 19  # cityscapes classes

batch_size = 2
epochs = 100
width = 512
height = 256
channels = 1

preprocess_mode = 'imagenet'  # the pretrained model has been preprocess with this method
name_best_model = args.weights

# build model and optimizer
model = Segception.Segception(num_classes=n_classes, weights=None, input_shape=(None, None, channels))

# Init models (optional, just for get_params function)
init_model(model, input_shape=(batch_size, width, height, channels))

# Init saver. can use also ckpt = tfe.Checkpoint((model=model, optimizer=optimizer,learning_rate=learning_rate, global_step=global_step)
restore_model = tfe.Saver(var_list=model.variables)

# restore if model saved and show number of params
restore_state(restore_model, name_best_model)

# Dictionary to map the cityscapes classes to cityscapes categories
class_to_category = {'0': 0, '1': 0, '2': 1, '3': 1, '4': 1, '5': 2, '6': 2, '7': 2, '8': 3, '9': 3, '10': 1, '11': 4,
                     '12': 4, '13': 5, '14': 5, '15': 5, '16': 5, '17': 5, '18': 5, '255': 255}


every_index = 5
# Get output dir name
out_dir = os.path.join(args.dataset, 'dataset_other_codification_10ms')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# dictionary which maps the export file with the frame subsequences that are usefull
export_file_dict = {os.path.join(args.dataset, 'run2', 'rec1487339175_export_other.hdf5'): [[0, 4150], [5200, 6600]],
                    os.path.join(args.dataset, 'run5', 'rec1487842276_export_other.hdf5'): [[1310, 1400], [1900, 2000], [2600, 3550]],
                    os.path.join(args.dataset, 'run3', 'rec1487593224_export_other.hdf5'): [[870, 2190]],
                    os.path.join(args.dataset, 'run5', 'rec1487846842_export_other.hdf5'): [[380, 500], [1800, 2150], [2575, 2730], [3530, 3900]],
                    os.path.join(args.dataset, 'run3', 'rec1487417411_export_other.hdf5'): [[100*every_index, 1500*every_index], [2150*every_index, 3100*every_index], [3200*every_index, 4430*every_index], [4840*every_index, 5150*every_index]],
                    os.path.join(args.dataset, 'run3', 'rec1487356509_export_other.hdf5'): [[2900, 3100]],
                    os.path.join(args.dataset, 'run4', 'rec1487779465_export_other.hdf5'): [[1800, 3400], [4000, 4700], [8400, 8630], [8800, 9160], [9920, 10175], [18500, 22300]]}


# contains the filename of the test subsequence
test_file = os.path.join(args.dataset, 'run3', 'rec1487417411_export_other.hdf5')
other_file = os.path.join(args.dataset, 'run3', 'rec1487356509_export_other.hdf5')

# Output directories
out_dir_images = os.path.join(out_dir, 'images')
out_dir_events = os.path.join(out_dir, 'events')
out_dir_labels = os.path.join(out_dir, 'labels')

if not os.path.exists(out_dir_images):
    os.makedirs(os.path.join(out_dir_images, 'train'))
    os.makedirs(os.path.join(out_dir_images, 'test'))
    os.makedirs(os.path.join(out_dir_images, 'other'))
if not os.path.exists(out_dir_events):
    os.makedirs(os.path.join(out_dir_events, 'train'))
    os.makedirs(os.path.join(out_dir_events, 'other'))
    os.makedirs(os.path.join(out_dir_events, 'test'))
if not os.path.exists(out_dir_labels):
    os.makedirs(os.path.join(out_dir_labels, 'train'))
    os.makedirs(os.path.join(out_dir_labels, 'other'))
    os.makedirs(os.path.join(out_dir_labels, 'test'))

# For every file to extract the data...
for filename in export_file_dict.keys():
    if os.path.isfile(filename):
        file_name = filename.split('/')[-1]
        name = file_name.replace('.hdf5', '')

        if filename in test_file:
            out_dir_images_folder = os.path.join(out_dir_images, 'test')
            out_dir_events_folder = os.path.join(out_dir_events, 'test')
            out_dir_labels_folder = os.path.join(out_dir_labels, 'test')
        elif filename in other_file:
            out_dir_images_folder = os.path.join(out_dir_images, 'other')
            out_dir_events_folder = os.path.join(out_dir_events, 'other')
            out_dir_labels_folder = os.path.join(out_dir_labels, 'other')
        else:
            out_dir_images_folder = os.path.join(out_dir_images, 'train')
            out_dir_events_folder = os.path.join(out_dir_events, 'train')
            out_dir_labels_folder = os.path.join(out_dir_labels, 'train')

        print ('Extracting ' + filename + '...')
        f = h5py.File(filename, 'r')

        ranges = export_file_dict[filename] # get the valid subsequence range

        for range in ranges:
            from_id =int(range[0])
            to_id = int(range[1])
            dvs_ = f['dvs_frame'][from_id:to_id, :, :, :]  # 6 event dimensions
            img_ = f['aps_frame'][from_id:to_id, :, :]  # 1 grayscale image  dimension
            if every_index > 1:
                step = every_index
            else:
                step = 1
            for i in xrange(0, img_.shape[0], step):
                events = dvs_[i, :, :, :] # get event information
                #check_timestamps(on, off)
                img = img_[i, :, :] # get iamge information
                img = np.expand_dims(img, 0)
                img = np.expand_dims(img, -1)
                pred_labels = inference(model, img, n_classes, flip_inference=True, scales=[0.75, 1, 1.5],
                                preprocess_mode=preprocess_mode)
                pred_labels = np.argmax(pred_labels, 3)
                pred_labels = np.array(transform(pred_labels[0, :, :], class_to_category))

                # postprocess labels 205-240 bottom pixels (Y axis) are noisy
                pred_labels_to_write = pred_labels[:200, :]
                img_to_write = img_[i,:200, :]
                events_to_write = events[:200, :, :]

                name_out = name + '_' + str(int(from_id + i)/step)
                # set output filenames
                out_img_name = os.path.join(out_dir_images_folder, name_out + '.png')
                out_label_name = os.path.join(out_dir_labels_folder, name_out + '.png')
                out_event_name = os.path.join(out_dir_events_folder, name_out)
                # save data
                cv2.imwrite(out_img_name, img_to_write)
                cv2.imwrite(out_label_name, pred_labels_to_write)
                np.save(out_event_name, events_to_write)

