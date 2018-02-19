from __future__ import division
from scipy.io.arff import loadarff
import logging
from pprint import pprint
from collections import defaultdict
from math import log

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GAIN_THRESHOLD = 1e-5

class DataSet(object):
    def __init__(self, data=None):
        self.data = data # nested list with last column as labels
        self.attr_types = {} # attribute # -> attr type

    def create(self, data, meta):
        attributes = []
        for _line in data:
            line = list(_line)
            attributes.append(list(line))
        self.data = attributes
        meta_types = meta.types()
        # Don't need meta information about class,
        # So exclude it
        for i,attr_type in enumerate(meta_types[:-1]):
            self.attr_types[i] = attr_type

    def get_attr_type(self, attr_number):
        return self.attr_types[attr_number]

    def get_feature(self, index):
        return [x[index] for x in self.features]


class Node(object):
    def __init__(self):
        self.children = []

    def split(self, dataset, feature, split_on):
        splits = defaultdict(list)
        data = dataset.data
        if dataset.get_attr_type(feature) == "nominal" :
            # for nominal splits, we do 1 split per value
            for row in data:
                for split in split_on:
                    if row[feature] == split:
                        splits[split].append(row)
        elif dataset.get_attr_type(feature) == "numeric" :
            # for numeric, we split at midpoint of data
            for row in data:
                if row[feature] <= split_on:
                    splits['left'].append(row)
                else:
                    splits['right'].append(row)
        else:
            logger.warn("Invalid Attribute to split on specified")
            return None
        return splits


    def get_split_points(self, dataset, feature):
        feature_values = get_column(dataset.data, feature)
        if dataset.get_attr_type(feature) == "nominal":
            return list(set(feature_values))
        elif dataset.get_attr_type(feature) == "numeric":
            return get_numeric_split_thresholds(feature_values)
        else:
            logger.warn("Invalid Attribute to split on specified")
            return None

    def build_tree(self, dataset):
        best_gain = 0.0
        best_feature = None
        best_splits = None
        data = dataset.data
        if len(data) == 0:
            return Node()
        current_entropy = entropy(data)
        # Identify best split
        for feature in range(len(data[0]) - 1):
            split_points = self.get_split_points(dataset, feature)
            if dataset.get_attr_type(feature) == "nominal":
                splits = self.split(dataset, feature, split_points)
                gain = calculate_info_gain(current_entropy, splits, dataset)
                logger.debug("Gain for Feature %s on split points %s : %f", feature, split_points, gain)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_splits = splits
            elif dataset.get_attr_type(feature) == "numeric":
                for split_point in split_points:
                    splits = self.split(dataset, feature, split_point)
                    gain = calculate_info_gain(current_entropy, splits, dataset)
                    logger.debug("Gain for Feature %s on split point %s : %f", feature, split_point, gain)
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature
                        best_splits = splits
        logger.info("Best split is on feature %s resulting in info gain of %f", best_feature, best_gain)
        if best_gain <= GAIN_THRESHOLD: # TODO - add more stopping criteria
            # at a leaf node
            # calculate impurity and return node
            pass
        else:
            # if not leaf, grow tree
            for split in best_splits.values():
                print "Building new tree with splits %s" % splits
                self.children.append(self.build_tree(DataSet(split)))

def calculate_info_gain(start_entropy, splits, dataset):
    gain = start_entropy
    total_len = len(dataset.data)
    for attr, split_set in splits.iteritems():
        this_split_ent = entropy(split_set)
        this_split_prob = len(split_set)/total_len
        gain -= this_split_ent * this_split_prob
    return gain

def get_numeric_split_thresholds(feature_data):
    split_candidates = []
    sorted_feature_data = sorted(feature_data)
    for a, b in zip(sorted_feature_data, sorted_feature_data[1:]):
        if a != b:
            split_candidates.append((a + b)/2)
    return split_candidates

def get_column(data, col):
    return [x[col] for x in data]

def entropy(data):
    ent = 0.0
    labels = get_column(data, -1)
    label_counts = {}
    possible_labels = set(labels)
    for label in possible_labels:
        label_counts[label] = labels.count(label)
    for label,label_count in label_counts.iteritems():
        p = float(label_count)/len(labels)
        ent -= p * log(p, 2)
    return ent

def parse_arf(file_handle):
    data, meta = loadarff(file_handle)
    return data, meta

if __name__ == "__main__":
    data, meta = parse_arf(file("credit_train.arff"))
    train_data = DataSet()
    train_data.create(data[::30], meta)
    n = Node()
    n.build_tree(train_data)
