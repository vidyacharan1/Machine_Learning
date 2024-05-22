from PIL import Image
import numpy as np
import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import time

# from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt
# from sklearn.feature_selection import SelectKBest, f_classif
# from sklearn.tree import DecisionTreeClassifier, plot_tree
# from sklearn.model_selection import GridSearchCV


# Define path of the train folder
train_path = '/home/vidya/Desktop/A3/train'
val_path = '/home/vidya/Desktop/A3/validation' 
test_realface_path = '/home/vidya/Desktop/A3/test_realface'

# Define an empty list to store the pixel values and labels of each image


# Loop through each image in the train folder and extract its pixel values
def load_images_from_folder(train_path):
    pixel_data = []
    labels = []
    for foldername in os.listdir(train_path):
        folder_path = os.path.join(train_path, foldername)
        # print(foldername)
        if not os.path.isdir(folder_path):
            continue
        for filename in os.listdir(folder_path):
            if not filename.endswith(('.png')):
                continue
            with Image.open(os.path.join(folder_path, filename)) as img:
                pixels = list(img.getdata())
                pixels2 = []
                for i in range(len(pixels)):
                    for j in range(len(pixels[0])):
                        pixels2.append(pixels[i][j])
                
                # Append the image's pixel values and label to the list
                pixel_data.append(pixels2)
                if foldername == 'person':
                    labels.append(1)
                elif foldername == 'car':
                    labels.append(0)
                elif foldername == 'airplane':
                    labels.append(2)
                elif foldername == 'dog':
                    labels.append(3)
                
    # Convert the list of pixel data and labels into a Pandas DataFrame
    df = pd.DataFrame(pixel_data)
    df['labels'] = labels
    return df

# Print the first five rows of the DataFrame
# print(df.head())

# train_size = 2000
# val_size = 400
def load_images_test(test_path):
    pixel_data = []
    # Loop through each image in the train folder and extract its pixel values
    for filename in os.listdir(test_path):
        if not filename.endswith(('.png')):
            continue
        with Image.open(os.path.join(test_path, filename)) as img:
            rgb_image = img.convert('RGB')
            pixels = list(rgb_image.getdata())
            pixels2 = []
            for i in range(len(pixels)):
                for j in range(len(pixels[0])):
                    pixels2.append(pixels[i][j])
            # Append the image's pixel values to the list
            pixel_data.append(pixels2)
    # print(len(pixels2))
    # Convert the list of pixel data into a Pandas DataFrame
    df = pd.DataFrame(pixel_data)
    return df 


train_data = load_images_from_folder(train_path).iloc[:, :]
val_data = load_images_from_folder(val_path).iloc[:, :]
test_realface_data = load_images_test(test_realface_path).iloc[:, :]

# pca = PCA(n_components=3072)
# test_realface_data = pca.fit_transform(test_realface)

# print(train_data.head())

# print(train_data.iloc[:,-1])

max_depth = 10
min_samples_split = 7
model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)

# Train the model and measure the training time
start_time = time.time()
model.fit(train_data.iloc[:, :-1], train_data.iloc[:, -1])
end_time = time.time()
train_time = end_time - start_time
print('The time taken to train : ',train_time)

# Make predictions on the training and validation sets
train_preds = model.predict(train_data.iloc[:, :-1])
val_preds = model.predict(val_data.iloc[:, :-1])
test_realface_preds = model.predict(test_realface_data.iloc[:, :])

# Compute the Accuracy, Precision, and Recall scores for the training and validation sets
train_acc = accuracy_score(train_data.iloc[:, -1], train_preds)
# train_prec = precision_score(train_data.iloc[:, -1], train_preds)
train_prec = precision_score(train_data.iloc[:, -1], train_preds, average='macro')
# train_rec = recall_score(train_data.iloc[:, -1], train_preds)
# train_rec = recall_score(train_data.iloc[:, -1], train_preds, average='macro')
train_rec = recall_score(train_data.iloc[:, -1], train_preds, average=None)

# val_acc = accuracy_score(val_data.iloc[:, -1], val_preds)
# # val_prec = precision_score(val_data.iloc[:, -1], val_preds)
# val_prec = precision_score(val_data.iloc[:, -1], val_preds, average='macro')
# # val_rec = recall_score(val_data.iloc[:, -1], val_preds)
# val_rec = recall_score(val_data.iloc[:, -1], val_preds, average=None)

actual_val = [1 for i in range(len(test_realface_preds))]
test_realface_acc = accuracy_score(actual_val, test_realface_preds)

print('The predictions for the faces :',test_realface_preds)
print('Accuracy :',test_realface_acc*100)

# Print the scores and training time
# print('Training Time:', train_time)
# print('Training Accuracy:', train_acc)
# print('Training Precision:', train_prec)
# print('Training Recall:', train_rec)

