import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import  matplotlib.patches as patches
def compute_iou(box_1, box_2):
    '''
    This function takes a pair of bounding boxes and returns intersection-over-
    union (IoU) of two bounding boxes.
    '''
    minx_inner = max(box_1[1],box_2[1])
    maxx_inner = min(box_1[3],box_2[3])
    
    miny_inner = max(box_1[0],box_2[0])
    maxy_inner = min(box_1[2],box_2[2])
    intersection = (maxx_inner - minx_inner+1) * (maxy_inner - miny_inner+1)
    union = (box_1[2] - box_1[0] +1) * (box_1[3] - box_1[1]+1) + (box_2[2] - box_2[0]+1) * (box_2[3] - box_2[1]+1) - intersection
    iou = intersection/union

    # print(iou, intersection, union,(maxx_inner - minx_inner+1),(maxy_inner - miny_inner+1))
    # print(box_1, box_2)
    iou = max(iou, 0)
    iou = min(iou, 1)
    assert (iou >= 0) and (iou <= 1.0)

    return iou


def compute_counts(preds, gts, iou_thr=0.5, conf_thr=0.5):
    '''
    This function takes a pair of dictionaries (with our JSON format; see ex.) 
    corresponding to predicted and ground truth bounding boxes for a collection
    of images and returns the number of true positives, false positives, and
    false negatives. 
    <preds> is a dictionary containing predicted bounding boxes and confidence
    scores for a collection of images.
    <gts> is a dictionary containing ground truth bounding boxes for a
    collection of images.
    '''
    TP = 0
    FP = 0
    FN = 0
    M = 0
    '''
    BEGIN YOUR CODE
    '''
    
    for pred_file in preds.keys():
        pred = preds[pred_file]
        gt = gts[pred_file]
        for i in range(len(gt)):
            ious = []
            for j in range(len(pred)):

                iou = compute_iou(pred[j][:4], gt[i])
                #print(pred[j][4])
                if pred[j][4]> conf_thr:
                    M += 1
                    ious.append(iou)
            positives = sum(np.array(ious) > iou_thr)
            if positives == 0:
                FN += 1
            elif positives == 1:
                TP += 1
            elif positives > 1:
                TP += 1
                FP += positives -1
                if FP < 10:
                    show_box(pred, pred[:][:4])
                
                
                
                
            #     if iou > iou_thr and pred[j][5] > conf_thr:
            #         TP += 1
            #     elif pred[j][5]> conf_thr:
            #         FP += 1
            #     else

            #     ious[i,j] = iou
            # positives = sum(ious[i,:] > iou_thr)
            # negatives = sum(ious[i,:] < 0)
            # if positives == 0:
            #     FN += 1
            # elif positives == 1:
            #     TP += 1
            # else:
            #     TP += 1
            #     FP += positives-1

                
    '''
    END YOUR CODE
    '''

    return TP, FP, FN

def show_box(key, boxs):
    #index = file_names_train.index(key)
    I = Image.open(os.path.join(data_path,key))
    I = np.asarray(I)
    img = mpimg.imread(data_path+'/'+key)

    for box in boxs:
        box_height = box[2] - box[0]
        box_width = box[3] - box[1]
        figure, ax = plt.subplots(1)
        rect = patches.Rectangle((box[0],box[1]),box_width,box_height, edgecolor='r', facecolor="none")
        ax.imshow(img)
        ax.add_patch(rect)

# set a path for predictions and annotations:
HOME = 'C:/Users/Jerem/OneDrive/Documents from one drive/GitHub/caltech-ee148-spring2020-hw02'

preds_path = HOME + '/data/hw02_preds'
gts_path = HOME + '/data/hw02_annotations'

# load splits:
split_path = HOME + '/data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True

'''
Load training data. 
'''
with open(os.path.join(preds_path,'preds_train.json'),'r') as f:
    preds_train = json.load(f)
    
with open(os.path.join(gts_path, 'annotations_train.json'),'r') as f:
    gts_train = json.load(f)

if done_tweaking:
    
    '''
    Load test data.
    '''
    
    with open(os.path.join(preds_path,'preds_test.json'),'r') as f:
        preds_test = json.load(f)
        
    with open(os.path.join(gts_path, 'annotations_test.json'),'r') as f:
        gts_test = json.load(f)


# For a fixed IoU threshold, vary the confidence thresholds.
# The code below gives an example on the training set for one IoU threshold. 


#confidence_thrs = np.sort(np.array([preds_train[fname][4] for fname in preds_train],dtype=float)) # using (ascending) list of confidence scores as thresholds
scores = []
for fname in preds_train:
    for box in preds_train[fname]:
        scores.append(box[4])
confidence_thrs = np.sort(np.array(scores))
tp_train = np.zeros(len(confidence_thrs))
fp_train = np.zeros(len(confidence_thrs))
fn_train = np.zeros(len(confidence_thrs))
for i, conf_thr in enumerate(confidence_thrs):
    tp_train[i], fp_train[i], fn_train[i] = compute_counts(preds_train, gts_train, iou_thr=0.5, conf_thr=conf_thr)

# Plot training set PR curves
M = fp_train + tp_train
N = fn_train + tp_train
P = tp_train/M
R  = tp_train/N
plt.plot(R,P)


if done_tweaking:
    print('Code for plotting test set PR curves.')
