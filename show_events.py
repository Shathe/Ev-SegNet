import numpy as np
import tensorflow as tf
import argparse
import cv2
import os.path
import glob
parser = argparse.ArgumentParser()
parser.add_argument("--eventfile", help="Events file to show",
                    default='/media/msrobot/discoGordo/Event-based/INIGO/dataset_other_codification/events/test')

parser.add_argument("--folder", help="folder to save events (iamges)",
                    default='/media/msrobot/discoGordo/Event-based/INIGO/dataset_other_codification/events_images_2')#rec1487417411_export_3765
args = parser.parse_args()

if not os.path.exists(args.folder):
    os.makedirs(args.folder)

if os.path.isdir(args.eventfile):
    for file_event in glob.glob(args.eventfile  + '/*'):
        name =file_event.split('/')[-1].replace('.npy', '')
        events = np.load(file_event)
        img_reduced =((events[:, :, 0] + 1) * 127).astype(np.uint8) + ((events[:, :, 1] + 1) * 127).astype(np.uint8)
        if events.shape[2] > 2:
            #img_reduced += (events[:, :, 3] * 255).astype(np.uint8)
            #img_reduced +=((events[:, :, 2] ) * 20).astype(np.uint8)
            pass
        if events.shape[2] > 4:
            img_reduced +=  (events[:, :, 5] * 255).astype(np.uint8)
            img_reduced += (events[:, :, 4]  * 20).astype(np.uint8)
        img_reduced = cv2.resize(img_reduced, (352, 224))

        cv2.imwrite(args.folder + '/' + name + '.png', img_reduced)

else:

    name = args.eventfile.split('/')[-1].replace('.npy','')
    events = np.load(args.eventfile)
    if events.shape[2] == 6:
        # Our codification
        cv2.imwrite(args.folder + '/' + name + '_events_on.png', ((events[:, :, 0] + 1) * 127).astype(np.uint8))
        cv2.imwrite(args.folder + '/' + name + '_events_off.png', ((events[:, :, 1] + 1) * 127).astype(np.uint8))
        cv2.imwrite(args.folder + '/' + name + '_events_on_mean.png', ((events[:, :, 2] + 1) * 127).astype(np.uint8))
        cv2.imwrite(args.folder + '/' + name + '_events_off_mean.png', ((events[:, :, 4] + 1) * 127).astype(np.uint8))
        cv2.imwrite(args.folder + '/' + name + '_events_on_std.png', (events[:, :, 3] * 255).astype(np.uint8))
        cv2.imwrite(args.folder + '/' + name + '_events_off_std.png', (events[:, :, 5] * 255).astype(np.uint8))
        img_reduced =((events[:, :, 0] + 1) * 127).astype(np.uint8) + ((events[:, :, 1] + 1) * 127).astype(np.uint8)
        img_reduced += (events[:, :, 3] * 255).astype(np.uint8) + (events[:, :, 5] * 255).astype(np.uint8)
        img_reduced +=((events[:, :, 2] ) * 20).astype(np.uint8) + ((events[:, :, 4] ) * 20).astype(np.uint8)
        img_reduced = cv2.resize(img_reduced, (352, 224))

        cv2.imwrite(args.folder + '/' + name + '_event_1_channel.png', img_reduced)


    else:
        # other codification
        cv2.imwrite(args.folder + '/' + name + '_events_on.png', ((events[:,:,0] + 1)* 127).astype(np.uint8))
        cv2.imwrite(args.folder + '/' + name + '_events_off.png', ((events[:,:,1] + 1)* 127).astype(np.uint8))
        cv2.imwrite(args.folder + '/' + name + '_events_on_recent.png', ((events[:,:,2] + 1)* 127).astype(np.uint8))
        cv2.imwrite(args.folder + '/' + name + '_events_off_recent.png', ((events[:,:,3] + 1)* 127).astype(np.uint8))
        mean_image = np.mean(events, axis = -1)
        cv2.imwrite(args.folder + '/' + name + '_mean_image.png', ((mean_image + 1) * 127).astype(np.uint8))
