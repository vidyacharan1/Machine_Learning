import os
from PIL import Image
import time
import numpy as np
import pandas as pd

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value = None, is_leaf = False):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.is_leaf = is_leaf

class DecisionTree:
    def __init__(self, max_depth = 10, min_samples_split = 7, criterion = 'gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth = 0):
        # Check if we have reached the stopping criteria
        if (depth >= self.max_depth or len(X) < self.min_samples_split or len(np.unique(y)) == 1):
            leaf_value = self._most_common_label(y)
            return Node(value = leaf_value, is_leaf = True)
        
        # Find the best split
        feature, threshold = self._best_split(X, y)
        
        # Split the data
        left_idx, right_idx = self._split(X[:, feature], threshold)
        left_X, left_y = X[left_idx, :], y[left_idx]
        right_X, right_y = X[right_idx, :], y[right_idx]
        
        # Recursively build the left and right subtrees
        left = self._build_tree(left_X, left_y, depth + 1)
        right = self._build_tree(right_X, right_y, depth + 1)
        
        # Return the node
        return Node(feature, threshold, left, right)

    def _best_split(self, X, y):
        best_gain = -1
        split_idx, split_threshold = None, None
        
        # For each feature find the best threshold
        for idx in range(X.shape[1]):
            # print(idx)
            X_column = X[:, idx]
            # print(X_column)
            thresholds = np.unique(X_column)
            
            # For each threshold find the information gain
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)
                
                if gain > best_gain:
                    best_gain = gain
                    split_idx = idx
                    split_threshold = threshold
                    
        return split_idx, split_threshold
    
    def _information_gain(self, y, X_column, split_threshold):
        # Parent loss
        parent_loss = self._gini(y)
        
        # Generate split
        left_idx, right_idx = self._split(X_column, split_threshold)
        if len(left_idx) == 0 or len(right_idx) == 0:
            return 0
        
        # Calculate the weighted loss
        n = len(y)
        n_l, n_r = len(left_idx), len(right_idx)
        e_l, e_r = self._gini(y[left_idx]), self._gini(y[right_idx])
        child_loss = (n_l / n) * e_l + (n_r / n) * e_r

        # Return information gain
        ig = parent_loss - child_loss
        return ig
    
    def _gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        return 1 - np.sum(np.square(probabilities))

    def _split(self, X_column, split_threshold):    
        left_idx = np.argwhere(X_column <= split_threshold).flatten()
        right_idx = np.argwhere(X_column > split_threshold).flatten()
        return left_idx, right_idx
    
    def _most_common_label(self, y):
        unique, counts = np.unique(y, return_counts=True)
        return unique[np.argmax(counts)]
    
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        if node.is_leaf:
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    
    def score(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.sum(y_pred == y) / len(y)
        return accuracy
    
    def precision(self, X, y):
        y_pred = self.predict(X)
        tp = 0
        fp = 0
        for i in range(len(y_pred)):
            if y_pred[i] == 1 and y[i] == 1:
                tp += 1
            elif y_pred[i] == 1 and y[i] == 0:
                fp += 1
        return tp/(tp+fp)
    
    def recall(self, X, y):
        y_pred = self.predict(X)
        tp = 0
        fn = 0
        for i in range(len(y_pred)):
            if y_pred[i] == 1 and y[i] == 1:
                tp += 1
            elif y_pred[i] == 0 and y[i] == 1:
                fn += 1
        return tp/(tp+fn)

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
# train_size = 2000
# val_size = 400
train_data = load_images_from_folder(train_path)
val_data = load_images_from_folder(val_path)

# Train the model
start_time = time.time()
model = DecisionTree()
model.fit(train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values)
end_time = time.time()

# Print the tree
# model.print_tree()

# Evaluate the model
print('Train Accuracy:', model.score(train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values))
print('Validation Accuracy:', model.score(val_data.iloc[:, :-1].values, val_data.iloc[:, -1].values))
print('Time taken to train the model:', end_time - start_time, 'seconds')
print('Train Precision:', model.precision(train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values))
print('Validation Precision:', model.precision(val_data.iloc[:, :-1].values, val_data.iloc[:, -1].values))
print('Train Recall:', model.recall(train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values))
print('Validation Recall:', model.recall(val_data.iloc[:, :-1].values, val_data.iloc[:, -1].values))








