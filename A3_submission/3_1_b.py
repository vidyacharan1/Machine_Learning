from PIL import Image
import numpy as np
import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import time
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree



# Define path of the train folder
train_path = '/home/vidya/Desktop/A3/train'
val_path = '/home/vidya/Desktop/A3/validation' 


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
                else:
                    labels.append(0)
    # print(labels)      
    # Convert the list of pixel data and labels into a Pandas DataFrame
    df = pd.DataFrame(pixel_data)
    df['labels'] = labels
    return df

# Print the first five rows of the DataFrame
# print(load_images_from_folder(train_path).head())
# print(load_images_from_folder(val_path).head())

# # Split the data into training and validation sets
train_size = 2000
val_size = 400
train_data = load_images_from_folder(train_path).iloc[:train_size, :]
val_data = load_images_from_folder(val_path).iloc[:val_size, :]
# print(val_data.iloc[:, -1].sum())
# val_data = df.iloc[train_size:, :]

# print(train_data.shape)
# Define the Decision Tree model with hyperparameters max depth and min samples split
max_depth = 10 #
min_samples_split = 7 #
model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split) #

# Train the model and measure the training time
start_time = time.time()
dt = model.fit(train_data.iloc[:, :-1], train_data.iloc[:, -1]) #
end_time = time.time()
train_time = end_time - start_time
print(train_time)

# Visualize Decision Tree
plt.figure(figsize=(10, 8))
plot_tree(dt, filled=True)
plt.show()

# Make predictions on the training and validation sets
train_preds = model.predict(train_data.iloc[:, :-1])
val_preds = model.predict(val_data.iloc[:, :-1]) #

# Compute the Accuracy, Precision, and Recall scores for the training and validation sets
train_acc = accuracy_score(train_data.iloc[:, -1], train_preds)
train_prec = precision_score(train_data.iloc[:, -1], train_preds)
train_rec = recall_score(train_data.iloc[:, -1], train_preds)

val_acc = accuracy_score(val_data.iloc[:, -1], val_preds)
val_prec = precision_score(val_data.iloc[:, -1], val_preds)
val_rec = recall_score(val_data.iloc[:, -1], val_preds)

# Print the scores and training time
print('Training Time:', train_time)
print('Training Accuracy:', train_acc)
print('Training Precision:', train_prec)
print('Training Recall:', train_rec)

print('Validation Accuracy:', val_acc)
print('Validation Precision:', val_prec)
print('Validation Recall:', val_rec)


y_pred = model.predict(val_data.iloc[:, :-1])
y_true = val_data.iloc[:, -1]
cm = confusion_matrix(y_true, y_pred)

fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()

