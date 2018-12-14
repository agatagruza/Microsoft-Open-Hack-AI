
# coding: utf-8

# In[146]:


import os
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator, load_img
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from keras.applications.vgg16 import VGG16

# Needed to display matplotlib plots in Jupyter
get_ipython().run_line_magic('matplotlib', 'inline')


# In[72]:


classifier = Sequential()
classifier.add(Conv2D(32, (6, 6), strides=(3, 3), input_shape=(128,128,3)))  # add relu?
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Conv2D(64, (6, 6), strides=(3, 3)))  # add relu?
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dense(32, activation='relu')) # add activation?
classifier.add(Flatten())
classifier.add(Dense(12, activation='softmax')) # add activation?
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[73]:


full_datagen = ImageDataGenerator(rescale = 1./255,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True,
                                 validation_split=0.3)


# In[74]:


imgdir = 'resize_images' # Folder containing extracted images
training_set = full_datagen.flow_from_directory(imgdir, target_size=(128,128),
                                            batch_size=32,
                                            class_mode="categorical",
                                           subset='training')
test_set = full_datagen.flow_from_directory(imgdir, target_size=(128,128),
                                            batch_size=32,
                                            class_mode="categorical",
                                           subset='validation')


# In[86]:


history = classifier.fit_generator(training_set, steps_per_epoch=160, epochs=15, validation_data=test_set, validation_steps=160)


# In[87]:


# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[136]:


imgdir = 'challenge3_test' # Folder containing images to check
categories = training_set.class_indices
categories = {v: k for k, v in categories.items()}
# Create a figure to display the images
fig = plt.figure(figsize=(12, 16))

# loop recursively through the folders
dir_num = 0
for root, folders, filenames in os.walk(imgdir):
    for imgFile in filenames:
        # in each folder, get the first file
        a=fig.add_subplot(1,5,dir_num + 1)
        filePath = os.path.join(root,imgFile)
        # Open it and add it to the figure (in a 4-row grid)
        img = Image.open(filePath)
        imgplot = plt.imshow(img)
        # Add the file name (the class of the image)
        im = load_img(filePath)
        prediction = classifier.predict(np.reshape(im,[1,128,128,3]))
        a.set_title(categories[np.argmax(prediction)])
        dir_num = dir_num + 1


# In[151]:


vgg = VGG16(weights='imagenet', include_top=False, input_shape=(128,128,3))
print(vgg.summary())


# In[149]:


print(classifier.summary())


# In[154]:


for layer in vgg.layers[:-2]:
    layer.trainable = False


# In[160]:


model = Sequential()

back_end = Sequential()
back_end.add(Flatten())
back_end.add(Dense(12, activation='softmax')) # add activation?

model.add(vgg)
model.add(back_end)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[161]:


print(model.summary())


# In[162]:


history_tl = model.fit_generator(training_set, steps_per_epoch=160, epochs=15, validation_data=test_set, validation_steps=160)


# In[164]:


# list all data in history
print(history_tl.history.keys())
# summarize history for accuracy
plt.plot(history_tl.history['acc'])
plt.plot(history_tl.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history_tl.history['loss'])
plt.plot(history_tl.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[163]:


imgdir = 'challenge3_test' # Folder containing images to check
categories = training_set.class_indices
categories = {v: k for k, v in categories.items()}
# Create a figure to display the images
fig = plt.figure(figsize=(12, 16))

# loop recursively through the folders
dir_num = 0
for root, folders, filenames in os.walk(imgdir):
    for imgFile in filenames:
        # in each folder, get the first file
        a=fig.add_subplot(1,5,dir_num + 1)
        filePath = os.path.join(root,imgFile)
        # Open it and add it to the figure (in a 4-row grid)
        img = Image.open(filePath)
        imgplot = plt.imshow(img)
        # Add the file name (the class of the image)
        im = load_img(filePath)
        prediction = model.predict(np.reshape(im,[1,128,128,3]))
        a.set_title(categories[np.argmax(prediction)])
        dir_num = dir_num + 1


# In[165]:


model.save('Challenge4.h5')

