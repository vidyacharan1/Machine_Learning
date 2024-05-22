import argparse 
import os
import cv2
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
            elif foldername == 'car':
                    labels.append(0)
            elif foldername == 'airplane':
                labels.append(2)
            elif foldername == 'dog':
                labels.append(3)
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



#part a 
model = DecisionTreeClassifier(max_depth=10, min_samples_split=7)
dt = model.fit(train_data.iloc[:, :-1], train_data.iloc[:, -1])
pred = dt.predict(test_data)


df = pd.DataFrame({'image_id': image_names, 'label': pred})
df.to_csv(os.path.join(out_folder, 'test_32a.csv'), index=False, header=False)



#part b
selector = SelectKBest(f_classif, k=10)
X_top10 = selector.fit_transform(train_data.iloc[:, :-1], train_data.iloc[:, -1])
mod = DecisionTreeClassifier(criterion='gini',max_depth=7, min_samples_split=4)
dec_tree = mod.fit(X_top10, train_data.iloc[:, -1])
X_top10_test = selector.transform(test_data)
pred = dec_tree.predict(X_top10_test)

df = pd.DataFrame({'image_id': image_names, 'label': pred})
df.to_csv(os.path.join(out_folder, 'test_32b.csv'), index=False, header=False)


#part c
ccp_alpha = 0.00468667
mode = DecisionTreeClassifier(random_state=38,ccp_alpha=ccp_alpha)
dec_tree = mode.fit(train_data.iloc[:, :-1], train_data.iloc[:, -1])
pred = dec_tree.predict(test_data)

df = pd.DataFrame({'image_id': image_names, 'label': pred})
df.to_csv(os.path.join(out_folder, 'test_32c.csv'), index=False, header=False)


#part d
modelll = RandomForestClassifier(random_state=25,criterion='entropy',max_depth=10, min_samples_split=10, n_estimators=150)
dec_tree = modelll.fit(train_data.iloc[:, :-1], train_data.iloc[:, -1])
pred = dec_tree.predict(test_data)

df = pd.DataFrame({'image_id': image_names, 'label': pred})
df.to_csv(os.path.join(out_folder, 'test_32d.csv'), index=False, header=False)


#part e
xgb = XGBClassifier(random_state=25, max_depth=7, n_estimators=50, subsample=0.6)
xgb.fit(train_data.iloc[:, :-1], train_data.iloc[:, -1])
pred = xgb.predict(test_data)

df = pd.DataFrame({'image_id': image_names, 'label': pred})
df.to_csv(os.path.join(out_folder, 'test_32e.csv'), index=False, header=False)



#part h 

xgb = XGBClassifier(random_state=25, max_depth=7, n_estimators=50, subsample=0.6)
xgb.fit(train_data.iloc[:, :-1], train_data.iloc[:, -1])
pred = xgb.predict(test_data)

df = pd.DataFrame({'image_id': image_names, 'label': pred})
df.to_csv(os.path.join(out_folder, 'test_32h.csv'), index=False, header=False)




