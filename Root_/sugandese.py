import os
import pandas as pd
import shutil


df = pd.read_csv('TableS1.csv', index_col='id') # open TableS1 as df
# df.set_index('id') # set unique picture id to be index for df
imageDir = os.listdir('./dataset') # get list of image files
count_instance = {'Ixodes scapularis':0, 'Amblyomma americanum':0, 'Dermacentor variabilis':0} 
# three indexes [dog, deer, and lone star]
# if none of the above
train = {'Ixodes scapularis':1200, 'Amblyomma americanum':300, 'Dermacentor variabilis':1200} # hopefully there are enough instances for this to be true...
# script to count instances...


for i in range(len(imageDir)): 
	index = int(imageDir[i].split('.')[0]) # first part of the file
	print(index)
	print(type(index))
	# ie the dataframe index
	debugVar = df.loc[index]['scientific_name']

	try:
		count_instance[df.loc[index]['scientific_name']] += 1

		if count_instance[df.loc[index]['scientific_name']] > train[df.loc[index]['scientific_name']]:
			# add to validation set
			shutil.copyfile('./dataset/'+imageDir[i], './new_dataset/validation/'+df.loc[index]['scientific_name']+'/'+imageDir[i])
		
		else:
			shutil.copyfile('./dataset/'+imageDir[i], './new_dataset/train/'+df.loc[index]['scientific_name']+'/'+imageDir[i])


	except:
		print('Fail: ', debugVar, count_instance)
	
	
	

print(count_instance)