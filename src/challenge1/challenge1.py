
# coding: utf-8

# In[1]:


get_ipython().system(' curl -O https://challenge.blob.core.windows.net/challengefiles/gear_images.zip')


# In[2]:


get_ipython().system(' unzip -o gear_images.zip')


# In[3]:


import os
import shutil
import numpy as np
import requests
import matplotlib.pyplot as plt
from PIL import Image

requests.get
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
        # Open it and add it to the figure (in a 4-row grid)
        img = Image.open(filePath)
        a=fig.add_subplot(4,np.ceil(len(folders)/4),dir_num + 1)
        imgplot = plt.imshow(img)
        
        # Add the folder name (the class of the image)
        a.set_title(folder)
        dir_num = dir_num + 1

