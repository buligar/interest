# """
# 2 ways to load HAM10000 dataset for skin cancer lesion classification
# Dataset link:
# https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000
# Data description:
# https://arxiv.org/ftp/arxiv/papers/1803/1803.10417.pdf
# The 7 classes of skin cancer lesions included in this dataset are:
# Melanocytic nevi (nv)
# Melanoma (mel)
# Benign keratosis-like lesions (bkl)
# Basal cell carcinoma (bcc)
# Actinic keratoses (akiec)
# Vascular lesions (vas)
# Dermatofibroma (df)
# """
#
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import os
# from glob import glob
# from PIL import Image
#
# ###########################################
# # METHOD 1: Read files using file name from the csv and add corresponding
# # image in a pandas dataframe along with labels.
# # This requires lot of memory to hold all thousands of images.
# # Use datagen if you run into memory issues.
#
# skin_df = pd.read_csv('HAM10000_metadata.csv')
#
# # Now time to read images based on image ID from the CSV file
# # This is the safest way to read images as it ensures the right image is read for the right ID
# image_path = {os.path.splitext(os.path.basename(x))[0]: x
#               for x in glob(os.path.join('input/archive/', '*', '*.jpg'))}
#
# # Define the path and add as a new column
# skin_df['path'] = skin_df['image_id'].map(image_path.get)
# # Use the path to read images.
# skin_df['image'] = skin_df['path'].map(lambda x: np.asarray(Image.open(x).resize((32, 32))))
#
# print(skin_df['dx'].value_counts())
#
# n_samples = 5  # number of samples for plotting
# # Plotting
# fig, m_axs = plt.subplots(7, n_samples, figsize=(4 * n_samples, 3 * 7))
# for n_axs, (type_name, type_rows) in zip(m_axs,
#                                          skin_df.sort_values(['dx']).groupby('dx')):
#     n_axs[0].set_title(type_name)
#     for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state=1234).iterrows()):
#         c_ax.imshow(c_row['image'])
#         c_ax.axis('off')
# #############################################################################
# # Reorganize data into subfolders based on their labels
# # then use keras flow_from_dir or pytorch ImageFolder to read images with
# # folder names as labels
#
# # Sort images to subfolders first
# import pandas as pd
# import os
# import shutil
#
# # Dump all images into a folder and specify the path:
# data_dir = os.getcwd() + "/input/HAM10000_all"
#
# # Path to destination directory where we want subfolders
# dest_dir = os.getcwd() + "/input/reorganized/"
#
# # Read the csv file containing image names and corresponding labels
# skin_df2 = pd.read_csv('HAM10000_metadata.csv')
# print(skin_df['dx'].value_counts())
#
# label = skin_df2['dx'].unique().tolist()  # Extract labels into a list
# label_images = []
#
# # Copy images to new folders
# for i in label:
#     os.mkdir(dest_dir + str(i) + "/")
#     sample = skin_df2[skin_df2['dx'] == i]['image_id']
#     label_images.extend(sample)
#     for id in label_images:
#         shutil.copyfile((data_dir + "/" + id + ".jpg"), (dest_dir + i + "/" + id + ".jpg"))
#     label_images = []
#
# # Now we are ready to work with images in subfolders

### FOR Keras datagen ##################################
# flow_from_directory Method
# useful when the images are sorted and placed in there respective class/label folders
# identifies classes automatically from the folder name.
# create a data generator

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import os
# import seaborn as sns
# from glob import glob
# from PIL import Image
# from keras.utils.np_utils import to_categorical  # used for converting labels to one-hot-encoding
# from sklearn.model_selection import train_test_split
# from scipy import stats
# from sklearn.preprocessing import LabelEncoder
# import autokeras as ak
# from keras.preprocessing.image import ImageDataGenerator
#
# np.random.seed(42)
# #
# # # Define datagen. Here we can define any transformations we want to apply to images
# # datagen = ImageDataGenerator()
# #
# # # define training directory that contains subfolders
# # train_dir = os.getcwd() + "/input/HAM10000/"
# # # USe flow_from_directory
# # train_data_keras = datagen.flow_from_directory(directory=train_dir,
# #                                                class_mode='categorical',
# #                                                batch_size=16,  # 16 images at a time
# #                                                target_size=(32, 32))  # Resize images
# #
# # # We can check images for a single batch.
# # x, y = next(train_data_keras)
# # # View each image
# # for i in range(0, 15):
# #     image = x[i].astype(int)
# #     plt.imshow(image)
# #     plt.show()
# #
# # # Now you can train via model.fit_generator
# #
# # ##################################################################################
# # ### Similarly FOR PYTORCH we can use DataLoader
# # import torchvision
# # from torchvision import transforms
# # import torch.utils.data as data
# # import numpy as np
# #
# # # Define root directory with subdirectories
# # train_dir = os.getcwd() + "/input/HAM10000/"
# #
# # # If you want to apply ransforms
# # TRANSFORM_IMG = transforms.Compose([
# #     transforms.Resize(32),
# #     transforms.RandomHorizontalFlip(),
# #     transforms.ToTensor(),  # Converts your input image to PyTorch tensor.
# #     transforms.Normalize(mean=[0.5, 0.5, 0.5],
# #                          std=[0.5, 0.5, 0.5])
# # ])
# #
# # # With transforms
# # # train_data = torchvision.datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=TRANSFORM_IMG)
# # # Without transforms
# # train_data_torch = torchvision.datasets.ImageFolder(root=train_dir)
# # # train_data_loader_torch = data.DataLoader(train_data_torch, batch_size=len(train_data_torch))
# #
# # print("Number of train samples: ", len(train_data_torch))
# # print("Detected Classes are: ", train_data_torch.class_to_idx)  # classes are detected by folder structure
# #
# # labels = np.array(train_data_torch.targets)
# # (unique, counts) = np.unique(labels, return_counts=True)
# # frequencies = np.asarray((unique, counts)).T
# # print(frequencies)
#
# ##########################################################################
# skin_df = pd.read_csv('HAM10000_metadata.csv')
#
# SIZE = 32
#
# # label encoding to numeric values from text
# le = LabelEncoder()
# le.fit(skin_df['dx'])
# LabelEncoder()
# print(list(le.classes_))
#
# skin_df['label'] = le.transform(skin_df["dx"])
# print(skin_df.sample(10))
#
# # Data distribution visualization
# fig = plt.figure(figsize=(15, 10))
#
# ax1 = fig.add_subplot(221)
# skin_df['dx'].value_counts().plot(kind='bar', ax=ax1)
# ax1.set_ylabel('Count')
# ax1.set_title('Cell Type');
#
# ax2 = fig.add_subplot(222)
# skin_df['sex'].value_counts().plot(kind='bar', ax=ax2)
# ax2.set_ylabel('Count', size=15)
# ax2.set_title('Sex');
#
# ax3 = fig.add_subplot(223)
# skin_df['localization'].value_counts().plot(kind='bar')
# ax3.set_ylabel('Count', size=12)
# ax3.set_title('Localization')
#
# ax4 = fig.add_subplot(224)
# sample_age = skin_df[pd.notnull(skin_df['age'])]
# sns.distplot(sample_age['age'], fit=stats.norm, color='red');
# ax4.set_title('Age')
#
# plt.tight_layout()
# plt.show()
#
# # Distribution of data into various classes
# from sklearn.utils import resample
#
# print(skin_df['label'].value_counts())
#
# # Balance data.
# # Many ways to balance data... you can also try assigning weights during model.fit
# # Separate each classes, resample, and combine back into single dataframe
#
# df_0 = skin_df[skin_df['label'] == 0]
# df_1 = skin_df[skin_df['label'] == 1]
# df_2 = skin_df[skin_df['label'] == 2]
# df_3 = skin_df[skin_df['label'] == 3]
# df_4 = skin_df[skin_df['label'] == 4]
# df_5 = skin_df[skin_df['label'] == 5]
# df_6 = skin_df[skin_df['label'] == 6]
#
# n_samples = 500
# df_0_balanced = resample(df_0, replace=True, n_samples=n_samples, random_state=42)
# df_1_balanced = resample(df_1, replace=True, n_samples=n_samples, random_state=42)
# df_2_balanced = resample(df_2, replace=True, n_samples=n_samples, random_state=42)
# df_3_balanced = resample(df_3, replace=True, n_samples=n_samples, random_state=42)
# df_4_balanced = resample(df_4, replace=True, n_samples=n_samples, random_state=42)
# df_5_balanced = resample(df_5, replace=True, n_samples=n_samples, random_state=42)
# df_6_balanced = resample(df_6, replace=True, n_samples=n_samples, random_state=42)
#
# skin_df_balanced = pd.concat([df_0_balanced, df_1_balanced,
#                               df_2_balanced, df_3_balanced,
#                               df_4_balanced, df_5_balanced, df_6_balanced])
#
# # Now time to read images based on image ID from the CSV file
# # This is the safest way to read images as it ensures the right image is read for the right ID
# print(skin_df_balanced['label'].value_counts())
#
# image_path = {os.path.splitext(os.path.basename(x))[0]: x
#               for x in glob(os.path.join('input/HAM10000/', '*', '*.jpg'))}
# # Define the path and add as a new column
# skin_df_balanced['path'] = skin_df['image_id'].map(image_path.get)
# # Use the path to read images.
# skin_df_balanced['image'] = skin_df_balanced['path'].map(lambda x: np.asarray(Image.open(x).resize((SIZE, SIZE))))
#
# n_samples = 5
#
# # Plot
# fig, m_axs = plt.subplots(7, n_samples, figsize=(4 * n_samples, 3 * 7))
# for n_axs, (type_name, type_rows) in zip(m_axs,
#                                          skin_df_balanced.sort_values(['dx']).groupby('dx')):
#     n_axs[0].set_title(type_name)
#     for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state=1234).iterrows()):
#         c_ax.imshow(c_row['image'])
#         c_ax.axis('off')
#
# # Convert dataframe column of images into numpy array
# X = np.asarray(skin_df_balanced['image'].tolist())
# X = X / 255.  # Scale values to 0-1. You can also used standardscaler or other scaling methods.
# Y = skin_df_balanced['label']  # Assign label values to Y
# Y_cat = to_categorical(Y, num_classes=7)  # Convert to categorical as this is a multiclass classification problem
# # Split to training and testing. Get a very small dataset for training as we will be
# # fitting it to many potential models.
# x_train_auto, x_test_auto, y_train_auto, y_test_auto = train_test_split(X, Y_cat, test_size=0.95, random_state=42)
#
# # Further split data into smaller size to get a small test dataset.
# x_unused, x_valid, y_unused, y_valid = train_test_split(x_test_auto, y_test_auto, test_size=0.05, random_state=42)
#
# # Define classifier for autokeras. Here we check 25 different models, each model 25 epochs
# clf = ak.ImageClassifier(max_trials=25)  # MaxTrials - max. number of keras models to try
# clf.fit(x_train_auto, y_train_auto, epochs=25)
#
# # Evaluate the classifier on test data
# _, acc = clf.evaluate(x_valid, y_valid)
# print("Accuracy = ", (acc * 100.0), "%")
#
# # get the final best performing model
# model = clf.export_model()
# print(model.summary())
#
# # Save the model
# model.save('cifar_model.h5')
#
# score = model.evaluate(x_valid, y_valid)
# print('Test accuracy:', score[1])

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import keras
import seaborn as sns
from PIL import Image
from glob import glob
from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical  # used for converting labels to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.preprocessing import LabelEncoder

skin_df = pd.read_csv('HAM10000_metadata.csv')
np.random.seed(42)
SIZE = 100

# label encoding to numeric values from text
le = LabelEncoder()
le.fit(skin_df['dx'])
LabelEncoder()
print(list(le.classes_))

skin_df['label'] = le.transform(skin_df["dx"])
print(skin_df.sample(10))

# Data distribution visualization
fig = plt.figure(figsize=(12, 8))

ax1 = fig.add_subplot(221)
skin_df['dx'].value_counts().plot(kind='bar', ax=ax1)
ax1.set_ylabel('Count')
ax1.set_title('Cell Type');

ax2 = fig.add_subplot(222)
skin_df['sex'].value_counts().plot(kind='bar', ax=ax2)
ax2.set_ylabel('Count', size=15)
ax2.set_title('Sex');

ax3 = fig.add_subplot(223)
skin_df['localization'].value_counts().plot(kind='bar')
ax3.set_ylabel('Count', size=12)
ax3.set_title('Localization')

ax4 = fig.add_subplot(224)
sample_age = skin_df[pd.notnull(skin_df['age'])]
sns.distplot(sample_age['age'], fit=stats.norm, color='red');
ax4.set_title('Age')

plt.tight_layout()
plt.show()

# Distribution of data into various classes
from sklearn.utils import resample

print(skin_df['label'].value_counts())

# Balance data.
# Many ways to balance data... you can also try assigning weights during model.fit
# Separate each classes, resample, and combine back into single dataframe

df_0 = skin_df[skin_df['label'] == 0]
df_1 = skin_df[skin_df['label'] == 1]
df_2 = skin_df[skin_df['label'] == 2]
df_3 = skin_df[skin_df['label'] == 3]
df_4 = skin_df[skin_df['label'] == 4]
df_5 = skin_df[skin_df['label'] == 5]
df_6 = skin_df[skin_df['label'] == 6]

n_samples = 500
df_0_balanced = resample(df_0, replace=True, n_samples=n_samples, random_state=42)
df_1_balanced = resample(df_1, replace=True, n_samples=n_samples, random_state=42)
df_2_balanced = resample(df_2, replace=True, n_samples=n_samples, random_state=42)
df_3_balanced = resample(df_3, replace=True, n_samples=n_samples, random_state=42)
df_4_balanced = resample(df_4, replace=True, n_samples=n_samples, random_state=42)
df_5_balanced = resample(df_5, replace=True, n_samples=n_samples, random_state=42)
df_6_balanced = resample(df_6, replace=True, n_samples=n_samples, random_state=42)

# Combined back to a single dataframe
skin_df_balanced = pd.concat([df_0_balanced, df_1_balanced,
                              df_2_balanced, df_3_balanced,
                              df_4_balanced, df_5_balanced, df_6_balanced])

# Check the distribution. All classes should be balanced now.
print(skin_df_balanced['label'].value_counts())

# Now time to read images based on image ID from the CSV file
# This is the safest way to read images as it ensures the right image is read for the right ID
image_path = {os.path.splitext(os.path.basename(x))[0]: x
              for x in glob(os.path.join('input/HAM10000/', '*', '*.jpg'))}

# Define the path and add as a new column
skin_df_balanced['path'] = skin_df['image_id'].map(image_path.get)
# Use the path to read images.
skin_df_balanced['image'] = skin_df_balanced['path'].map(lambda x: np.asarray(Image.open(x).resize((SIZE, SIZE))))
print()

n_samples = 5  # number of samples for plotting
# Plotting
fig, m_axs = plt.subplots(7, n_samples, figsize=(4 * n_samples, 3 * 7))
for n_axs, (type_name, type_rows) in zip(m_axs,
                                         skin_df_balanced.sort_values(['dx']).groupby('dx')):
    n_axs[0].set_title(type_name)
    for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state=1234).iterrows()):
        c_ax.imshow(c_row['image'])
        c_ax.axis('off')

# Convert dataframe column of images into numpy array
X = np.asarray(skin_df_balanced['image'].tolist())
X = X / 255.  # Scale values to 0-1. You can also used standardscaler or other scaling methods.
Y = skin_df_balanced['label']  # Assign label values to Y
Y_cat = to_categorical(Y, num_classes=7)  # Convert to categorical as this is a multiclass classification problem
# Split to training and testing
x_train, x_test, y_train, y_test = train_test_split(X, Y_cat, test_size=0.25, random_state=42)

# Define the model.
# I've used autokeras to find out the best model for this problem.
# You can also load pretrained networks such as mobilenet or VGG16

num_classes = 7

model = Sequential()
model.add(Conv2D(256, (3, 3), activation="relu", input_shape=(SIZE, SIZE, 3)))
# model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(128, (3, 3), activation='relu'))
# model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model.add(Flatten())

model.add(Dense(32))
model.add(Dense(7, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['acc'])

# Train
# You can also use generator to use augmentation during training.

batch_size = 32
epochs = 100

history = model.fit(
    x_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(x_test, y_test),
    verbose=2)

score = model.evaluate(x_test, y_test)
print('Test accuracy:', score[1])

# plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Prediction on test data
y_pred = model.predict(x_test)
# Convert predictions classes to one hot vectors
y_pred_classes = np.argmax(y_pred, axis=1)
# Convert test data to one hot vectors
y_true = np.argmax(y_test, axis=1)

# Print confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)

fig, ax = plt.subplots(figsize=(6, 6))
sns.set(font_scale=1.6)
sns.heatmap(cm, annot=True, linewidths=.5, ax=ax)

# PLot fractional incorrect misclassifications
incorr_fraction = 1 - np.diag(cm) / np.sum(cm, axis=1)
plt.bar(np.arange(7), incorr_fraction)
plt.xlabel('True Label')
plt.ylabel('Fraction of incorrect predictions')