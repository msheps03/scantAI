import os
import pandas as pd
import shutil


df = pd.read_csv('TableS1.csv') # open TableS1 as df
# df.set_index('id') # set unique picture id to be index for df
imageDir = os.listdir('./dataset') # get list of image files
count_instance = {'Ixodes scapularis':0, 'Amblyomma americanum':0, 'Dermacentor variabilis':0} 
# three indexes [dog, deer, and lone star]
# if none of the above
train = {'Ixodes scapularis':1000, 'Amblyomma americanum':300, 'Dermacentor variabilis':1000} # hopefully there are enough instances for this to be true...
# script to count instances...


for i in range(len(imageDir)): 
    index = imageDir[i].split('.')[0] # first part of the file
    # ie the dataframe index
    debugVar = df.loc[i]['scientific_name']


    count_instance[df.loc[i]['scientific_name']] += 1

    if count_instance[df.loc[i]['scientific_name']] > train[df.loc[i]['scientific_name']]:
        # add to validation set
        shutil.copyfile('./dataset/'+imageDir[i], './new_dataset/validation/'+df.loc[i]['scientific_name']+imageDir[i])

    else:
        shutil.copyfile('./dataset/'+imageDir[i], './new_dataset/train/'+df.loc[i]['scientific_name']+imageDir[i])

    
    print('Fail: ', index, count_instance)
	
	
	

print(count_instance)