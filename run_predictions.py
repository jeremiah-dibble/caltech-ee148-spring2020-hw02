import os
import numpy as np
import json
from PIL import Image

def compute_convolution(I, T, stride=None):
    '''
    This function takes an image <I> and a template <T> (both numpy arrays) 
    and returns a heatmap where each grid represents the output produced by 
    convolution at each location. You can add optional parameters (e.g. stride, 
    window_size, padding) to create additional functionality. 
    '''
    (n_rows,n_cols,n_channels) = np.shape(I)

    '''
    BEGIN YOUR CODE
    '''
    stride = 1
    trows, tcols, tchannels = np.shape(T)
    heat_rows = n_rows - trows
    heat_cols = n_cols - tcols
    heatmap = np.zeros((int(heat_rows/stride), int(heat_cols/stride)))
    sub_heat = 0
    col  = 0
    row =  0
    
    # rows = np.arange(0,n_rows, stride)
    # cols = np.arange(0,n_cols, stride)
    # for row in rows:
    #     for col in cols:
            
   # print(heat_rows)
    while (row < (heat_rows-1)):
        col = 0   
        while col < heat_cols-stride:

            for channel in range(n_channels):
                sub_I = I[row:(row+trows),col:col+tcols, channel]
                #print(np.shape(sub_I))
                # print(np.shape(T))
                channel_T = T[:,:,channel]
                #print(np.convolve(sub_I.reshape((-1,)),channel_T.reshape((-1,)),'valid'))
                #sub_heat += np.sum(np.multiply(channel_T,sub_I))/(np.linalg.norm(T)*np.linalg.norm(sub_I))
                sub_heat += np.convolve(sub_I.reshape((-1,)),channel_T.reshape((-1,)),'valid')

            heatmap[int(row/stride),int(col/stride)] = sub_heat
            sub_heat = 0
            col += stride
        row += 1
        
        
    '''
    END YOUR CODE
    ''' 
    # print('min max is', np.min(heatmap))
    # print('max', np.max(heatmap))
    # print(heatmap)
    #img = Image.fromarray(heatmap)
        
    #img.show()
    return heatmap


def predict_boxes(heatmap, template_D):
    '''
    This function takes heatmap and returns the bounding boxes and associated
    confidence scores.
    '''

    output = []

    '''
    BEGIN YOUR CODE
    '''
    
    '''
    As an example, here's code that generates between 1 and 5 random boxes
    of fixed size and returns the results in the proper format.
    '''
    stride = 10
    candidates = []
    template_height, template_width =template_D
    n_rows, n_cols = np.shape(heatmap)
    thirds = np.linspace(0,n_cols, 4)
    sub_sections  = [heatmap[:,0:int(thirds[1])], heatmap[:,int(thirds[1]):int(thirds[2])],heatmap[:,int(thirds[2]):]]
    normal_std = np.std(heatmap/np.max(heatmap))
    #print(np.max(heatmap))
    for i in range(len(thirds)-1 ):
        sub_heat = sub_sections[i]
        #print(np.shape(sub_heat))
        third_max = 0
        for row in range(n_rows):
            row_max = np.max(sub_heat[row])
            if row_max > third_max:
                third_max = row_max
                max_row = row
        location_row = max_row
        location_column = thirds[i] + sub_heat[max_row].tolist().index(third_max)
        candidates.append((location_row,location_column))
        #print(candidates, np.max(heatmap))
    #print(np.sum(heatmap > .98))    
    #print(np.sum(heatmap > 0))
    #print(np.mean(heatmap))
    for can in candidates:
        tl_row, tl_col = can
        tl_row, tl_col =int(tl_row), int(tl_col)
        

        #print('mean',np.mean(heatmap))
        
        score = heatmap[tl_row, tl_col] / np.max(heatmap)
        #print(score)
        tl_row = tl_row*10
        tl_col = tl_col * 10
        br_row = tl_row + template_height
        br_col = tl_col + template_width
        #print('this score', score)
        #print('score cutoff', 1 + normal_std)
        #print(heatmap[tl_row, tl_col])
        if True:#score > (np.max(heatmap) + normal_std):
            output.append([tl_row,tl_col,br_row,br_col, score])
        

    '''
    END YOUR CODE
    '''

    return output


def detect_red_light_mf(I):
    '''
    This function takes a numpy array <I> and returns a list <output>.
    The length of <output> is the number of bounding boxes predicted for <I>. 
    Each entry of <output> is a list <[row_TL,col_TL,row_BR,col_BR,score]>. 
    The first four entries are four integers specifying a bounding box 
    (the row and column index of the top left corner and the row and column 
    index of the bottom right corner).
    <score> is a confidence score ranging from 0 to 1. 

    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''

    '''
    BEGIN YOUR CODE
    '''
    template_height = 8
    template_width = 6
    
    temp_file = file_names_train[np.random.randint(0,len(file_names_train))]
    while len(gts_train[temp_file]) < 1:
        temp_file = file_names_train[np.random.randint(0,len(file_names_train))]
        print('image has no boxes')
    tl_row,tl_col,br_row,br_col =gts_train[temp_file][np.random.randint(0,len(gts_train[temp_file]))] 
    template_height =  br_row - tl_row
    template_width =   br_col - tl_col                    
    # You may use multiple stages and combine the results
    
    T = Image.open(os.path.join(data_path,temp_file))
    

    # convert to numpy array:
    T = np.asarray(T)
    T = T[int(tl_row):int(br_row),int(tl_col):int(br_col),:]

    trows, tcols, tchannel = np.shape(T)
    heatmap = compute_convolution(I, T)
    output = predict_boxes(heatmap, (template_height, template_width))

    '''
    END YOUR CODE
    '''

    for i in range(len(output)):
        assert len(output[i]) == 5
        assert (output[i][4] >= 0.0) and (output[i][4] <= 1.0)

    return output

# Note that you are not allowed to use test data for training.
# set the path to the downloaded data:
home = 'C:/Users/Jerem/OneDrive/Documents from one drive/GitHub/caltech-ee148-spring2020-hw02'    
data_path = home+'/data/RedLights2011_Medium'

# load splits: 
split_path = home+'/data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# set a path for saving predictions:
preds_path = home+'/data/hw02_preds'
os.makedirs(preds_path, exist_ok=True) # create directory if needed

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True

'''
Make predictions on the training set.
'''
preds_train = {}
print(len(file_names_train))
for i in range(len(file_names_train)):
    print(i/len(file_names_train))
    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names_train[i]))

    # convert to numpy array:
    I = np.asarray(I)

    preds_train[file_names_train[i]] = detect_red_light_mf(I)
    

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds_train.json'),'w') as f:
    json.dump(preds_train,f)

if done_tweaking:
    '''
    Make predictions on the test set. 
    '''
    preds_test = {}
    for i in range(len(file_names_test)):

        # read image using PIL:
        I = Image.open(os.path.join(data_path,file_names_test[i]))

        # convert to numpy array:
        I = np.asarray(I)

        preds_test[file_names_test[i]] = detect_red_light_mf(I)

    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path,'preds_test.json'),'w') as f:
        json.dump(preds_test,f)
