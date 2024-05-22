import argparse 
import os
import cv2
import numpy as np
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier 

parser = argparse.ArgumentParser()
parser.add_argument("--train_path")
parser.add_argument("--test_path")
parser.add_argument("--out_path")
args = parser.parse_args()

train_folder = args.train_path
test_folder = args.test_path
out_folder = args.out_path

    

def load_images_from_folder(train_path):
    pixel_data = []
    labels = []
    for foldername in os.listdir(train_path):
        folder_path = os.path.join(train_path, foldername)
        if not os.path.isdir(folder_path):
            continue
        for filename in os.listdir(folder_path):
            if not filename.endswith(('.png')):
                continue
            img = cv2.imread(os.path.join(folder_path, filename))
            pixels = img.flatten().tolist()
            pixel_data.append(pixels)
            if foldername == 'person':
                labels.append(1)
            else:
                labels.append(0)
    df = pd.DataFrame(pixel_data)
    df['labels'] = labels
    return df

train_data = load_images_from_folder(train_folder).iloc[:, :]

def load_images_test(test_path):
    pixel_data = []
    image_ids = []
    for filename in os.listdir(test_path):
        if not filename.endswith(('.png')):
            continue
        img = cv2.imread(os.path.join(test_path, filename))
        pixels = img.flatten().tolist()
        pixel_data.append(pixels)
        #remove the .png from the filename
        filename = filename[:-4]
        image_ids.append(filename)
    df = pd.DataFrame(pixel_data)
    df['image_id'] = image_ids
    return df

test_data = load_images_test(test_folder).iloc[:, :-1]
image_names = load_images_test(test_folder).iloc[:, -1]


# part a
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

model = DecisionTree()
model.fit(train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values)
pred = model.predict(test_data.values)

df = pd.DataFrame({'image_id': image_names, 'label': pred})
df.to_csv(os.path.join(out_folder, 'test_31a.csv'), index=False, header=False)
    

#part b 
model = DecisionTreeClassifier(max_depth=10, min_samples_split=7)
dt = model.fit(train_data.iloc[:, :-1], train_data.iloc[:, -1])
pred = dt.predict(test_data)


df = pd.DataFrame({'image_id': image_names, 'label': pred})
df.to_csv(os.path.join(out_folder, 'test_31b.csv'), index=False, header=False)


#part c
selector = SelectKBest(f_classif, k=10)
X_top10 = selector.fit_transform(train_data.iloc[:, :-1], train_data.iloc[:, -1])
mod = DecisionTreeClassifier(criterion='entropy',max_depth=5, min_samples_split=7)
dec_tree = mod.fit(X_top10, train_data.iloc[:, -1])
X_top10_test = selector.transform(test_data)
pred = dec_tree.predict(X_top10_test)

df = pd.DataFrame({'image_id': image_names, 'label': pred})
df.to_csv(os.path.join(out_folder, 'test_31c.csv'), index=False, header=False)



#part d
ccp_alpha = 0.00378205
mode = DecisionTreeClassifier(random_state=38,ccp_alpha=ccp_alpha)
dec_tree = mode.fit(train_data.iloc[:, :-1], train_data.iloc[:, -1])
pred = dec_tree.predict(test_data)

df = pd.DataFrame({'image_id': image_names, 'label': pred})
df.to_csv(os.path.join(out_folder, 'test_31d.csv'), index=False, header=False)


#part e
modelll = RandomForestClassifier(random_state=25,criterion='entropy',max_depth=10, min_samples_split=5, n_estimators=150)
dec_tree = modelll.fit(train_data.iloc[:, :-1], train_data.iloc[:, -1])
pred = dec_tree.predict(test_data)

df = pd.DataFrame({'image_id': image_names, 'label': pred})
df.to_csv(os.path.join(out_folder, 'test_31e.csv'), index=False, header=False)


#part f
xgb = XGBClassifier(random_state=25, max_depth=10, n_estimators=50, subsample=0.6)
xgb.fit(train_data.iloc[:, :-1], train_data.iloc[:, -1])
pred = xgb.predict(test_data)

df = pd.DataFrame({'image_id': image_names, 'label': pred})
df.to_csv(os.path.join(out_folder, 'test_31f.csv'), index=False, header=False)


#part h 
xgb = XGBClassifier(random_state=25, max_depth=10, n_estimators=50, subsample=0.6)
xgb.fit(train_data.iloc[:, :-1], train_data.iloc[:, -1])
pred = xgb.predict(test_data)

df = pd.DataFrame({'image_id': image_names, 'label': pred})
df.to_csv(os.path.join(out_folder, 'test_31h.csv'), index=False, header=False)

