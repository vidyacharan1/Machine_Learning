from PIL import Image
import numpy as np
import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import time
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV



# Define path of the train folder
train_path = '/home/vidya/Desktop/A3/train'
val_path = '/home/vidya/Desktop/A3/validation' 
test_path = '/home/vidya/Desktop/A3/test_sample'

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

    # print(labels)      
    # Convert the list of pixel data and labels into a Pandas DataFrame
    df = pd.DataFrame(pixel_data)
    df['labels'] = labels
    return df

# Print the first five rows of the DataFrame
# print(load_images_from_folder(train_path).head())
# print(load_images_from_folder(val_path).head())

# Split the data into training and validation sets
train_size = 2000
val_size = 400
train_data = load_images_from_folder(train_path).iloc[:train_size, :]
val_data = load_images_from_folder(val_path).iloc[:val_size, :]

#With default hyperparameters

# # create a random forest classifier with default hyperparameters
rfc = RandomForestClassifier(random_state=38)
rfc.fit(train_data.iloc[:, :-1], train_data.iloc[:, -1])

# # predict the labels of training and validation data
# train_preds = rfc.predict(train_data.iloc[:, :-1])
# val_preds = rfc.predict(val_data.iloc[:, :-1])

# # calculate the accuracy, precision, and recall of training and validation data
# train_acc = accuracy_score(train_data.iloc[:, -1], train_preds)
# train_prec = precision_score(train_data.iloc[:, -1], train_preds)
# train_rec = recall_score(train_data.iloc[:, -1], train_preds)

# val_acc = accuracy_score(val_data.iloc[:, -1], val_preds)
# val_prec = precision_score(val_data.iloc[:, -1], val_preds)
# val_rec = recall_score(val_data.iloc[:, -1], val_preds)

# print("Default hyperparameters:")
# print(f"Training accuracy: {train_acc}")
# print(f"Training precision: {train_prec}")
# print(f"Training recall: {train_rec}")
# print(f"Validation accuracy: {val_acc}")
# print(f"Validation precision: {val_prec}")
# print(f"Validation recall: {val_rec}\n")

# With different hyperparameters

# define the hyperparameters grid
param_grid = {
    'n_estimators': [80, 100, 150, 200],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 7, 10],
    'min_samples_split': [5, 7, 10]
}

# perform grid search with cross-validation
grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5, n_jobs=-1) 
start_time = time.time()
grid_search.fit(train_data.iloc[:, :-1], train_data.iloc[:, -1])
end_time = time.time()

# report the best hyperparameters and their performance
best_params = grid_search.best_params_
print("Best hyperparameters:", best_params)

best_rfc = RandomForestClassifier(random_state=38, **best_params)
best_rfc.fit(train_data.iloc[:, :-1], train_data.iloc[:, -1])

train_preds = best_rfc.predict(train_data.iloc[:, :-1])
val_preds = best_rfc.predict(val_data.iloc[:, :-1])

train_acc = accuracy_score(train_data.iloc[:, -1], train_preds)
# train_prec = precision_score(train_data.iloc[:, -1], train_preds)
train_prec = precision_score(train_data.iloc[:, -1], train_preds, average='macro')
# train_rec = recall_score(train_data.iloc[:, -1], train_preds)
train_rec_macro = recall_score(train_data.iloc[:, -1], train_preds, average='macro')
train_rec = recall_score(train_data.iloc[:, -1], train_preds, average=None)

val_acc = accuracy_score(val_data.iloc[:, -1], val_preds)
# val_prec = precision_score(val_data.iloc[:, -1], val_preds)
val_prec = precision_score(val_data.iloc[:, -1], val_preds, average='macro')
val_rec_macro = recall_score(val_data.iloc[:, -1], val_preds, average='macro')
val_rec = recall_score(val_data.iloc[:, -1], val_preds, average=None)

print("\nBest hyperparameters performance:")
print('Training Accuracy:', train_acc)
print('Training Precision:', train_prec)
print('Training Recall using macro:', train_rec_macro)
print('Training Recall:', train_rec)

print('Validation Accuracy:', val_acc)
print('Validation Precision:', val_prec)
print('Validation Recall using macro:', val_rec_macro)
print('Validation Recall:', val_rec)
print('Time taken to execute:', end_time - start_time)

y_pred = val_preds
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