import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm
from typing import List, Tuple

"""
Notice:
    1) You can't add any additional package
    2) You can add or remove any function "except" fit, _build_tree, predict
    3) You can ignore the suggested data type if you want
"""

class ConvNet(nn.Module): # Don't change this part!
    def __init__(self):
        super(ConvNet, self).__init__()
        self.model = timm.create_model('mobilenetv3_small_100', pretrained=True, num_classes=300)

    def forward(self, x):
        x = self.model(x)
        return x
    
    
class DecisionTree:
    def __init__(self, max_depth=9):
        self.max_depth = max_depth

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.data_size = X.shape[0]
        total_steps = 2 ** self.max_depth
        self.progress = tqdm(total=total_steps, desc="Growing tree", position=0, leave=True)
        self.tree = self._build_tree(X, y, 0)
        self.progress.close()


    def _build_tree(self, X: pd.DataFrame, y: np.ndarray, depth: int):
        # (TODO) Grow the decision tree and return it
        # terminate
        if depth >= self.max_depth or len(np.unique(y)) == 1 or X.shape[0] < 2:
            return {'label': int(np.bincount(y).argmax())}
        best_feature_idx, best_threshold = DecisionTree._best_split(X, y)
        # if no best feature
        if best_feature_idx is None:
            return {'label': int(np.bincount(y).argmax())}
        # split data
        left_X, left_y, right_X, right_y = DecisionTree._split_data(X, y, best_feature_idx, best_threshold)
        # if no split
        if left_X.shape[0] == 0 or right_X.shape[0] == 0:
            return {'label': int(np.bincount(y).argmax())}
        
        # create node
        node = {
            'feature_index': best_feature_idx,
            'threshold': best_threshold,
            'left': self._build_tree(left_X, left_y, depth + 1),
            'right': self._build_tree(right_X, right_y, depth + 1)
        }
        self.progress.update(1)
        return node
    # def predict(self, X: pd.DataFrame)->torch.Tensor:
    #     # (TODO) Call _predict_tree to traverse the decision tree to return the classes of the testing dataset
    #     predictions = []
    #     for _, row in X.iterrows():
    #         predictions.append(self._predict_tree(row, self.tree))
    #     return torch.tensor(predictions, dtype=torch.long)
    def predict(self, X: pd.DataFrame)-> np.ndarray:
        # (TODO) Call _predict_tree to traverse the decision tree to return the classes of the testing dataset
        predictions = []
        for _, row in X.iterrows():
            prediction = self._predict_tree(row, self.tree)
            predictions.append(prediction)
        return np.array(predictions)

    def _predict_tree(self, x, tree_node):
        # (TODO) Recursive function to traverse the decision tree
        # lead node
        if 'label' in tree_node:
            return tree_node['label']
        # split data
        feature_index = tree_node['feature_index']
        threshold = tree_node['threshold']
        # left or right
        if x.iloc[feature_index] <= threshold:
            return self._predict_tree(x, tree_node['left'])
        else:
            return self._predict_tree(x, tree_node['right'])

    def _split_data(X: pd.DataFrame, y: np.ndarray, feature_index: int, threshold: float):
        # (TODO) split one node into left and right node 
        
        data = X.values  # convert to Numpy
        feature_val = data[:, feature_index]
        mask = feature_val <= threshold
        left_dataset_X = pd.DataFrame(data[mask], columns=X.columns).reset_index(drop=True)
        right_dataset_X = pd.DataFrame(data[~mask], columns=X.columns).reset_index(drop=True)
        left_dataset_y = y[mask]
        right_dataset_y = y[~mask]
        return left_dataset_X, left_dataset_y, right_dataset_X, right_dataset_y

    def _best_split(X: pd.DataFrame, y: np.ndarray):
        # (TODO) Use Information Gain to find the best split for a dataset
        data = X.values  # convert to Numpy
        origin_entropy = DecisionTree._entropy(y)
        best_info_gain = 0
        best_feature_index = None
        best_threshold = None
        feature_sz = X.shape[1]
        for i in range(feature_sz):
            feature_values = np.unique(data[:, i])
            # if all values are the same, skip this feature
            if len(feature_values) <= 1:
                continue

            thr_candidates = (feature_values[:-1] + feature_values[1:]) / 2
            for threshold in thr_candidates:
                # 直接對nparray型別做split_data
                mask = data[:, i] <= threshold
                left_y = y[mask]
                right_y = y[~mask]
                # calculate entropy
                left_entropy = DecisionTree._entropy(left_y)
                right_entropy = DecisionTree._entropy(right_y)
                # calculate information gain
                info_gain = origin_entropy - (len(left_y) / len(y)) * left_entropy - (len(right_y) / len(y)) * right_entropy
                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_feature_index = i
                    best_threshold = threshold
            
        return best_feature_index, best_threshold

    def _entropy(y: np.ndarray)->float:
        # (TODO) Return the entropy
         _, counts = np.unique(y, return_counts=True)
         probs = counts / counts.sum()
         entropy = 0.0
        # 累加每個類別的貢獻，避免 log2(0)
         for p in probs:
            if p > 0:
                entropy += -p * np.log2(p)
         return entropy
        

def get_features_and_labels(model: ConvNet, dataloader: DataLoader, device)->Tuple[List, List]:
    # (TODO) Use the model to extract features from the dataloader, return the features and labels
    model.eval()
    feats, labs = [], []
    with torch.no_grad():
        for imgs, lbl in tqdm(dataloader, desc="Extracting features & labels"):
            imgs = imgs.to(device)
            out = model(imgs).cpu().numpy()
            feats.extend(out)
            labs.extend(lbl.numpy())
    return pd.DataFrame(feats), np.array(labs)

def get_features_and_paths(model: ConvNet, dataloader: DataLoader, device)->Tuple[List, List]:
    # (TODO) Use the model to extract features from the dataloader, return the features and path of the images
    model.eval()
    feats, paths = [], []
    with torch.no_grad():
        for imgs, pths in tqdm(dataloader, desc="Extracting features & paths"):
            imgs = imgs.to(device)
            out = model(imgs).cpu().numpy()
            feats.extend(out)
            paths.extend(pths)
    return pd.DataFrame(feats), paths




#Validation Accuracy: 0.7820 for depth 7
#Validation Accuracy: 0.7893 for depth 5
#Validation Accuracy: 0.7959 for depth 9