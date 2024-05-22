from PIL import Image
import numpy as np
import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score, precision_score, recall_score
import time
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV


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
# # Split the data into training and validation sets
train_size = 2000
val_size = 400
train_data = load_images_from_folder(train_path).iloc[:train_size, :]
val_data = load_images_from_folder(val_path).iloc[:val_size, :]
# print(val_data.iloc[:, -1].sum())

selector = SelectKBest(f_classif, k=10)
X_top10 = selector.fit_transform(train_data.iloc[:, :-1], train_data.iloc[:, -1])
# X_top10_val = selector.fit_transform(val_data.iloc[:, :-1], val_data.iloc[:, -1])
X_top10_val = selector.transform(val_data.iloc[:, :-1])

# print(selector.get_support(indices=True))

# Build Decision Tree using top-10 features
dt = DecisionTreeClassifier()
dt.fit(X_top10, train_data.iloc[:, -1])

dtval = DecisionTreeClassifier()
dtval.fit(X_top10_val, val_data.iloc[:, -1])

# Visualize Decision Tree
# plt.figure(figsize=(10, 8))
# plot_tree(dt, filled=True)
# plt.show()

# Define hyperparameter space
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 7, 10, 15],
    'min_samples_split': [2, 4, 7, 9]
}

# Perform grid search over hyperparameter space
grid_search = GridSearchCV(dt, param_grid, cv=5)
start_time = time.time()
grid_search.fit(X_top10, train_data.iloc[:, -1])
end_time = time.time()

grid_val_search = GridSearchCV(dtval, param_grid, cv=5)
grid_val_search.fit(X_top10_val, val_data.iloc[:, -1])

# Print best hyperparameters and corresponding accuracy
print("Best hyperparameters for train data: ", grid_search.best_params_)
print("Training accuracy with best hyperparameters: ", grid_search.best_score_)
# print("Best hyperparameters for validation data: ", grid_val_search.best_params_)
print("Validation accuracy with best hyperparameters: ", grid_val_search.best_score_)
print("Time taken to train using best hyperparameters: ", end_time - start_time)

dt = DecisionTreeClassifier(criterion='gini', max_depth=7, min_samples_split=4)
dt.fit(X_top10, train_data.iloc[:, -1])
# Visualize Decision Tree
plt.figure(figsize=(10, 8))
plot_tree(dt , filled=True)
plt.show()

y_pred = grid_search.predict(X_top10_val)
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