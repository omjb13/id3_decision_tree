from scipy.io.arff import loadarff
import numpy as np
import pandas as pd
import logging
from collections import defaultdict

logger = logging.getLogger('DecisionTrees')
logger.setLevel(logging.DEBUG)

class DataSet(object):
    def __init__(self):
        # Matrix containing all attrs
        self.data = None
        # Vector of labels
        self.labels = None
        # Types for each attribute
        # Either nominal or numeric
        self.attr_types = {}

    def create(self, data, meta):
        attributes = []
        classes = []
        for _line in data:
            line = list(_line)
            attributes.append(list(line[0:-1]))
            classes.append(line[-1])
        # going with Pandas dataframes
        # because they're easier to work with
        # when it comes to mixed types
        self.data = pd.DataFrame(attributes)
        self.data.columns = range(1, len(attributes[0]) + 1)
        self.labels = pd.DataFrame(classes)
        # let's parse the meta information
        meta_types = meta.types()
        # the last value is class, which we don't need
        for i,attr_type in enumerate(meta_types[:-1]):
            self.attr_types[i+1] = attr_type

    def get_attr_type(self, attr_number):
        return self.attr_types[attr_number]


class Node(object):
    def __init__(self):
        self.children = []

    def create_tree(self, dataset):
        # identify possible candidate splits
        # evaluate each split
        # split on best split
        pass

    def split(self, dataset, feature):
        '''
        Splits on given feature
        :param dataset: object of type DataSet
        :param feature: Int specifying attr to split on
        :return: splits dict with split_value -> PD DF
        '''
        splits = defaultdict(list)
        feature_data = dataset.data[feature]
        if dataset.get_attr_type(feature) == "nominal" :
            # for nominal splits, we do 1 split per value
            to_split = set(feature_data)
            print to_split
            for row in dataset.data.itertuples():
                print row[feature]
                for split in to_split:
                    if row[feature] == split:
                        splits[split].append(row)
        # convert list of named tuples to PD DataFrame
            for split in to_split:
                splits[split] = pd.DataFrame.from_records(splits[split])
        elif dataset.get_attr_type(feature) == "numeric" :
            # for numeric, we split at midpoint of data
            sorted_feature_data = sorted(feature_data)
            print sorted_feature_data
            to_split = (sorted_feature_data[0]) + int(sorted_feature_data[-1])/2
            print to_split
            for row in dataset.data.itertuples():
                if row[feature] < to_split:
                    splits['left'].append(row)
                else:
                    splits['right'].append(row)
            for split in ['left', 'right']:
                splits[split] = pd.DataFrame.from_records(splits[split])
        else:
            logger.warn("Invalid Attribute to split on specified")
            return None
        return splits


def parse_arf(file_handle):
    data, meta = loadarff(file_handle)
    return data, meta

if __name__ == "__main__":
    data, meta = parse_arf(file("credit_train.arff"))
    train_data = DataSet()
    train_data.create(data, meta)
    n = Node()
    print n.split(train_data, 2)
