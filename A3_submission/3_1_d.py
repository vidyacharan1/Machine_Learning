from PIL import Image
import numpy as np
import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score, precision_score, recall_score
import time
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
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

# Split the data into training and validation sets
train_size = 2000
val_size = 400
train_data = load_images_from_folder(train_path).iloc[:train_size, :]
val_data = load_images_from_folder(val_path).iloc[:val_size, :]





# Create a decision tree classifier object
dt = DecisionTreeClassifier(random_state=38)

# Fit the decision tree to the training data
dt.fit(train_data.iloc[:,:-1], train_data.iloc[:,-1])

# Get the pruning path and effective alphas
path = dt.cost_complexity_pruning_path(train_data.iloc[:,:-1], train_data.iloc[:,-1])
ccp_alphas = path.ccp_alphas
impurities = path.impurities

# print(ccp_alphas)
# print(impurities)


# Plot the total impurity vs ccp alpha

fig, ax = plt.subplots()
ax.plot(ccp_alphas, impurities, marker='o', drawstyle="steps-post")
ax.set_xlabel("Effective alpha")
ax.set_ylabel("Total impurity of leaves")
ax.set_title("Total Impurity vs Effective Alpha")
plt.show()

# create decision tree classifier objects for different values of ccp alpha
d_tree = []
for ccp_alpha in ccp_alphas:
    dt = DecisionTreeClassifier(random_state=38, ccp_alpha=ccp_alpha)
    dt.fit(train_data.iloc[:,:-1], train_data.iloc[:,-1])
    d_tree.append(dt)
           

# number of nodes
# num_nodes = [dt.tree_.node_count for dt in d_tree]
# tot_depth = [dt.tree_.max_depth for dt in d_tree]

# Plot the number of nodes vs ccp alpha

# fig, ax = plt.subplots()
# ax.plot(ccp_alphas, num_nodes, marker='o', drawstyle="steps-post")
# ax.set_xlabel("Effective alpha")
# ax.set_ylabel("Number of nodes")
# ax.set_title("Number of Nodes vs Effective Alpha")
# plt.show()

# Plot the depth vs ccp alpha

# fig, ax = plt.subplots()
# ax.plot(ccp_alphas, tot_depth, marker='o', drawstyle="steps-post")
# ax.set_xlabel("Effective alpha")
# ax.set_ylabel("Depth of tree")
# ax.set_title("Depth vs Effective Alpha")
# plt.show()


# Create empty lists to store accuracies
train_accs = [dt.score(train_data.iloc[:,:-1], train_data.iloc[:,-1]) for dt in d_tree]
val_accs = [dt.score(val_data.iloc[:,:-1], val_data.iloc[:,-1]) for dt in d_tree]




# Plot the accuracies vs ccp alpha
fig, ax = plt.subplots()
ax.plot(ccp_alphas, train_accs, marker='o', label='Training')
ax.plot(ccp_alphas, val_accs, marker='o', label='Validation')
ax.set_xlabel("Effective alpha")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy vs Effective Alpha")
ax.legend()
plt.show()


# Train the best tree on the entire training set
start_time = time.time()
ccp_alpha_grid_search = GridSearchCV(estimator=DecisionTreeClassifier(random_state=38), param_grid={'ccp_alpha': ccp_alphas}, cv=5)
ccp_alpha_grid_search.fit(train_data.iloc[:,:-1], train_data.iloc[:,-1])
end_time = time.time()
best_para = ccp_alpha_grid_search.best_params_
dt = ccp_alpha_grid_search.best_estimator_

# Evaluate the best tree on the training and validation set
train_acc = dt.score(train_data.iloc[:,:-1], train_data.iloc[:,-1])
val_acc = dt.score(val_data.iloc[:,:-1], val_data.iloc[:,-1])
print("Time taken to train the best tree:", end_time - start_time)
print("Best ccp_accp_alpha_grid_search.best_params_lpha:", best_para)
print("Training accuracy:", train_acc)
print("Validation accuracy:", val_acc)


# Visualize the best-pruned tree
plt.figure(figsize=(12,12))
plot_tree(dt, filled=True, class_names=['0', '1'])
plt.show()

# Print the confusion matrix
y_pred = dt.predict(val_data.iloc[:,:-1])
cm = confusion_matrix(val_data.iloc[:,-1], y_pred)

fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()





