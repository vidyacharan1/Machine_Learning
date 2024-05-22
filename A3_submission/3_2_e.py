from PIL import Image
import numpy as np
import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import time
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from xgboost.sklearn import XGBClassifier 



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


# Define parameter grids for grid search
gb_param_grid = {'n_estimators': [20, 30, 40, 50],
                 'subsample': [0.2, 0.3, 0.4, 0.5, 0.6],
                 'max_depth': [5, 6, 7, 8, 9, 10]}
xgb_param_grid = {'n_estimators': [20, 30, 40, 50],
                  'subsample': [0.2, 0.3, 0.4, 0.5, 0.6],
                  'max_depth': [5, 6, 7, 8, 9, 10]}

# Perform grid search for Gradient Boosting Classifier

gb_clf = GradientBoostingClassifier(random_state=38)
gb_grid_search = GridSearchCV(gb_clf, gb_param_grid, cv=5, n_jobs=-1)

start_time_gb = time.time()
gb_grid_search.fit(train_data.iloc[:, :-1], train_data.iloc[:, -1])
end_time_gb = time.time()

best_params = gb_grid_search.best_params_
dt = gb_grid_search.best_estimator_
print("Best hyperparameters for gradient boost:", best_params)

# Perform grid search for XGBoost Classifier

# xgb_clf = XGBClassifier(seed=25)
# xgb_grid_search = GridSearchCV(xgb_clf, xgb_param_grid, cv=5, n_jobs=-1)

# start_time_xgb = time.time()
# xgb_grid_search.fit(train_data.iloc[:, :-1], train_data.iloc[:, -1])
# end_time_xgb = time.time()

# best_params = xgb_grid_search.best_params_
# dt = xgb_grid_search.best_estimator_
# print("Best hyperparameters for XGboost:", best_params)

# Print results for Gradient Boosting Classifier

# print("Gradient Boosting Classifier:")
# print("Best parameters:", gb_grid_search.best_params_)
# print("Training accuracy:", accuracy_score(train_data.iloc[:, -1], gb_grid_search.predict(train_data.iloc[:, :-1])))
# print("Validation accuracy:", accuracy_score(val_data.iloc[:, -1], gb_grid_search.predict(val_data.iloc[:, :-1])))
# print("Training precision:", precision_score(train_data.iloc[:, -1], gb_grid_search.predict(train_data.iloc[:, :-1]), average='weighted'))
# print("Validation precision:", precision_score(val_data.iloc[:, -1], gb_grid_search.predict(val_data.iloc[:, :-1]), average='weighted'))
# print("Training recall:", recall_score(train_data.iloc[:, -1], gb_grid_search.predict(train_data.iloc[:, :-1]), average='weighted'))
# print("Validation recall:", recall_score(val_data.iloc[:, -1], gb_grid_search.predict(val_data.iloc[:, :-1]), average='weighted'))
# # print("Time taken:", gb_grid_search.cv_results_['mean_fit_time'].sum())
# print("Time taken for gb to execute:", end_time_gb - start_time_gb)

# Print results for XGBoost Classifier

# print("\nXGBoost Classifier:")
# print("Best parameters:", xgb_grid_search.best_params_)
# print("Training accuracy:", accuracy_score(train_data.iloc[:, -1], xgb_grid_search.predict(train_data.iloc[:, :-1])))
# print("Validation accuracy:", accuracy_score(val_data.iloc[:, -1], xgb_grid_search.predict(val_data.iloc[:, :-1])))
# print("Training precision:", precision_score(train_data.iloc[:, -1], xgb_grid_search.predict(train_data.iloc[:, :-1]), average='weighted'))
# print("Validation precision:", precision_score(val_data.iloc[:, -1], xgb_grid_search.predict(val_data.iloc[:, :-1]), average='weighted'))
# print("Training recall:", recall_score(train_data.iloc[:, -1], xgb_grid_search.predict(train_data.iloc[:, :-1]), average='weighted'))
# print("Validation recall:", recall_score(val_data.iloc[:, -1], xgb_grid_search.predict(val_data.iloc[:, :-1]), average='weighted'))
# # print("Time taken:", xgb_grid_search.cv_results_['mean_fit_time'].sum())
# print("Time taken for xgb to execute:", end_time_xgb - start_time_xgb)

y_pred = dt.predict(val_data.iloc[:, :-1])
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
