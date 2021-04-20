import numpy as np
import os
import simplejson as json
HOME = 'C:/Users/Jerem/OneDrive/Documents from one drive/GitHub/caltech-ee148-spring2020-hw02'
np.random.seed(2020) # to ensure you always get the same train/test split
preds_path = HOME + '/data/hw02_preds'
data_path = HOME +'/data/RedLights2011_Medium'
gts_path = HOME +  '/data/hw02_annotations'
split_path = HOME + '/data/hw02_splits'
os.makedirs(preds_path, exist_ok=True) # create directory if needed

split_test = True # set to True and run when annotations are available

train_frac = 0.85

# get sorted list of files:
file_names = sorted(os.listdir(data_path))

# remove any non-JPEG files:
file_names = [f for f in file_names if '.jpg' in f]

# split file names into train and test
file_names_train = []
file_names_test = []
'''
Your code below. 
'''

end_index = int(train_frac*len(file_names))
order = np.arange(0,len(file_names),1)
np.random.shuffle(order)
i = 0
for index in order:
    if i < end_index:
        file_names_train.append(file_names[index])
    else:
        file_names_test.append(file_names[index])
    i += 1

assert (len(file_names_train) + len(file_names_test)) == len(file_names)
assert len(np.intersect1d(file_names_train,file_names_test)) == 0

np.save(os.path.join(split_path,'file_names_train.npy'),file_names_train)
np.save(os.path.join(split_path,'file_names_test.npy'),file_names_test)

if split_test:
    with open(os.path.join(gts_path, 'formatted_annotations_students.json'),'r') as f:
        gts = json.load(f)
    
    # Use file_names_train and file_names_test to apply the split to the
    # annotations
    gts_train = {}
    gts_test = {}
    '''
    Your code below. 
    '''
    for file in file_names_train:
        gts_train[file] = gts[file]
    for file in file_names_test:
        gts_test[file] = gts[file]
    
    with open(os.path.join(gts_path, 'annotations_train.json'),'w') as f:
        json.dump(gts_train,f)
    
    with open(os.path.join(gts_path, 'annotations_test.json'),'w') as f:
        json.dump(gts_test,f)
    
    
