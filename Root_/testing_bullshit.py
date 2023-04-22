import os

path = "C:/Users/ShepDaddy/Pictures/scant_backgrounds/"

print(path.split('/')[-2])
dirList = []
for x in os.listdir(path):
    if x.endswith(".jpg"):
        # Prints only text file present in My Folder
        dirList.append(path+x)

print(dirList)

