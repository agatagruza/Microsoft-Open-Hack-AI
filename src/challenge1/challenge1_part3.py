
# coding: utf-8

# In[47]:


import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Needed to display matplotlib plots in Jupyter
get_ipython().run_line_magic('matplotlib', 'inline')

imgdir = 'gear_images' # Folder containing extracted images

# Create a figure to display the images
fig = plt.figure(figsize=(12, 16))

# loop recursively through the folders
dir_num = 0
for root, folders, filenames in os.walk(imgdir):
    for folder in folders:
        # in each folder, get the first file
        imgFile = os.listdir(os.path.join(root,folder))[0]
        filePath = os.path.join(root,folder,imgFile)
        print("root " + root + " folder " + folder)
        # Open it and add it to the figure (in a 4-row grid)
        img = Image.open(filePath)
        a=fig.add_subplot(4,np.ceil(len(folders)/4),dir_num + 1)
        img.thumbnail((128,128))
        imgNew = Image.new("RGB", (128,128), "white")
        offsetX = int((128 - img.width)/2)
        offsetY = int((128 - img.height)/2)
        imgNew.paste(img, (offsetX, offsetY))
        imgplot = plt.imshow(imgNew)
        # Add the folder name (the class of the image)
        a.set_title(folder)
        dir_num = dir_num + 1


# In[48]:


os.mkdir("resize_images")


# In[71]:


imgdir = 'gear_images' # Folder containing extracted images

# Create a figure to display the images
#fig = plt.figure(figsize=(12, 16))

# loop recursively through the folders
dir_num = 0
for root, folders, filenames in os.walk(imgdir):
    for folder in folders:
        print("folder " + folder)
        for imgFile in os.listdir(os.path.join(root, folder)):
            # in each folder, get the first file
            filePath = os.path.join(root,folder,imgFile)
            print(filePath)
            # Open it and add it to the figure (in a 4-row grid)
            img = Image.open(filePath)
            #a=fig.add_subplot(4,np.ceil(len(folders)/4),dir_num + 1)
            img.thumbnail((128,128))
            imgNew = Image.new("RGB", (128,128), "white")
            offsetX = int((128 - img.width)/2)
            offsetY = int((128 - img.height)/2)
            imgNew.paste(img, (offsetX, offsetY))
            resizePath = os.path.join("resize_images", folder)
            if not os.path.exists(resizePath):
                os.mkdir(resizePath)

            imgNew.save(os.path.join(resizePath, imgFile))
            #imgplot = plt.imshow(imgNew)
            # Add the folder name (the class of the image)
            #a.set_title(folder)
            #dir_num = dir_num + 1


# In[73]:


imgdir = 'gear_images' # Folder containing extracted images
resizeDir = 'resize_images'

# Create a figure to display the images
fig = plt.figure(figsize=(12, 50))

# loop recursively through the folders
dir_num = 0
for root, folders, filenames in os.walk(imgdir):
    for folder in folders:
        # in each folder, get the first file
        imgFile = os.listdir(os.path.join(root,folder))[0]
        filePath = os.path.join(root,folder,imgFile)
        # Open it and add it to the figure (in a 4-row grid)
        resize_img = Image.open(os.path.join(resizeDir, folder, imgFile))
        img = Image.open(filePath)
        
        a=fig.add_subplot(12,2,dir_num + 1)
        imgplot = plt.imshow(img)
        # Add the folder name (the class of the image)
        a.set_title(folder)
        
        a=fig.add_subplot(12,2,dir_num + 2)
        imgplot = plt.imshow(resize_img)
        a.set_title("resize_" + folder)

        dir_num = dir_num + 2

