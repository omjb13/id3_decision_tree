from __future__ import division
from scipy.io.arff import loadarff
import logging
from collections import defaultdict
from math import log
import operator
from sys import argv

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


class Node(object):
    def __init__(self,
                 split_on=None,
                 final_label=None,
                 label_counts=None,
                 is_leaf=False,
                 parent=None,
                 split_value=None):
        self.children = []
        # tuple of (split_attribute, split_point)
        self.split_on = split_on
        self.final_label = final_label
        self.label_counts = label_counts
        self.is_leaf = is_leaf
        self.parent = parent
        self.split_value = split_value


def build_tree(node, data):
    best_gain = 0.0
    best_feature = None
    best_splits = None
    new_best = False
    # Identify best split
    current_entropy = entropy(data)
    for feature in range(len(data[0]) - 1):
        split_points = get_split_points(data, feature)
        if get_attr_type(feature) == "nominal":
            splits = split_on_point(data, feature, split_points)
            gain = calculate_info_gain(current_entropy, splits, data)
            logger.debug("Gain for Feature %s on split points %s : %f", feature, split_points, gain)
            if gain > best_gain:
                new_best = True
            # tie resolution
            elif gain == best_gain and best_feature:
                if feature < best_feature[0]:
                    new_best = True
            if new_best:
                best_gain = gain
                best_feature = (feature, split_points)
                best_splits = splits
                new_best = False
        elif get_attr_type(feature) == "numeric":
            for split_point in split_points:
                splits = split_on_point(data, feature, split_point)
                gain = calculate_info_gain(current_entropy, splits, data)
                logger.debug("Gain for Feature %s on split point %s : %f", feature, split_point, gain)
                if gain > best_gain:
                    new_best = True
                elif gain == best_gain and best_feature:
                    if feature < best_feature[0]:
                        new_best = True
                    elif feature == best_feature[0]:
                        if split_point < best_feature[1]:
                            new_best = True
                if new_best:
                    best_gain = gain
                    best_feature = (feature, split_point)
                    best_splits = splits
                    new_best = False
    # BEGIN building the tree
    if best_gain <= GAIN_THRESHOLD or len(data) < MAX_ELEMENTS_IN_NODE \
            or split_points == [] or is_pure(data):  # TODO - add more stopping criteria
        # at a leaf node
        logger.info("Leaf node created with label counts %s", get_label_counts(data))
        node.is_leaf = True
        node.label_counts = get_label_counts(data)
        final_label = get_majority_label(data)
        if final_label == None:
            node.final_label = node.parent.final_label
        else:
            node.final_label = final_label
    else:
        # if not leaf, grow tree
        split_values = get_split_values(best_splits, best_feature)
        for split_value in split_values:
            # make a new node
            split = best_splits.setdefault(split_value, [[]])
            new_node = Node(split_on=best_feature,
                            label_counts=get_label_counts(split),
                            split_value=split_value)
            new_node.parent = node
            new_node.final_label = get_majority_label(split)
            logger.info("New node created with label counts %s", get_label_counts(split))
            # add it to it's parents children list
            node.children.append(new_node)
            build_tree(new_node, split)


def split_on_point(data, feature, split_on):
    splits = defaultdict(list)
    if get_attr_type(feature) == "nominal":
        for row in data:
            for split in split_on:
                if row[feature] == split:
                    splits[split].append(row)
    elif get_attr_type(feature) == "numeric":
        for row in data:
            if row[feature] <= split_on:
                splits['left'].append(row)
            else:
                splits['right'].append(row)
    else:
        logger.warn("Invalid Attribute to split on specified")
        return None
    return splits


def predict(node, record):
    while not node.is_leaf:
        for child in node.children:
            feature, split_point = child.split_on
            branch_value = child.split_value
            record_value = record[feature]
            if get_attr_type(feature) == "nominal":
                if branch_value == record_value:
                    node = child
            else:
                if record_value <= split_point and branch_value == "left":
                    node = child
                elif record_value > split_point and branch_value == "right":
                    node = child
    return node.final_label


## HELPER FUNCTIONS ##

def get_split_points(data, feature):
    feature_values = get_column(data, feature)
    if get_attr_type(feature) == "nominal":
        return list(set(feature_values))
    elif get_attr_type(feature) == "numeric":
        return get_numeric_split_thresholds(feature_values)
    else:
        logger.warn("Invalid Attribute to split on specified")
        return None


def get_majority_label(data):
    if data == [[]]:
        return None
    label_counts = get_label_counts(data)
    if label_counts.setdefault("+", 0) == label_counts.setdefault("-", 0):
        return None
    return max(label_counts.iteritems(), key=operator.itemgetter(1))[0]


def is_pure(data):
    return len(set(get_column(data, -1))) == 1


def get_split_values(best_splits, best_feature):
    # return a list of split_values in the order
    # they should be appended to the node
    feature, _ = best_feature
    type = get_attr_type(feature)
    if type == "nominal":
        value_list = meta._attributes[get_feature_name(feature)][1]
    elif type == "numeric":
        value_list = ["left", "right"]
    else:
        logger.warn("Invalid Feature Type")
    return value_list


def get_label_counts(data):
    label_counts = {}
    if data == [[]]:
        return label_counts
    labels = get_column(data, -1)
    possible_labels = set(labels)
    for label in possible_labels:
        label_counts[label] = labels.count(label)
    return label_counts


def calculate_info_gain(start_entropy, splits, data):
    gain = start_entropy
    total_len = len(data)
    for attr, split_set in splits.iteritems():
        this_split_ent = entropy(split_set)
        this_split_prob = len(split_set) / total_len
        gain -= this_split_ent * this_split_prob
    return gain


def get_numeric_split_thresholds(feature_data):
    split_candidates = []
    sorted_feature_data = sorted(feature_data)
    for a, b in zip(sorted_feature_data, sorted_feature_data[1:]):
        split_candidates.append((a + b) / 2)
    return split_candidates


def get_column(data, col):
    return [x[col] for x in data]


def get_feature_name(feature_index):
    return ATTR_NAMES[feature_index]


def entropy(data):
    ent = 0.0
    label_counts = get_label_counts(data)
    for label, label_count in label_counts.iteritems():
        p = float(label_count) / len(data)
        ent -= p * log(p, 2)
    return ent


def parse_arf(file_handle):
    data, meta = loadarff(file_handle)
    return data, meta


def build_datatype_dict(meta):
    attr_types = {}
    meta_types = meta.types()
    # Don't need meta information about class,
    # So exclude it
    for i, attr_type in enumerate(meta_types[:-1]):
        attr_types[i] = attr_type
    return attr_types


def print_tree(node, depth=-1):
    # prints a tree given the root node
    if node.split_on is not None:
        feature, split_point = node.split_on
        split_value = node.split_value
        to_print = ""
        to_print += "|\t" * depth
        to_print += get_feature_name(feature).lower()
        if get_attr_type(feature) == "nominal":
            to_print += " = "
            to_print += split_value.strip()
        else:
            if split_value == "left":
                to_print += " <= "
            else:
                to_print += " > "
            to_print += "%.6f" % split_point
        to_print += prettify_label_counts(node.label_counts)
        if node.is_leaf:
            to_print += ": " + node.final_label
        print to_print
    for child in node.children:
        print_tree(child, depth + 1)


def prettify_label_counts(label_counts):
    return " [%d %d]" % (label_counts.setdefault("+", 0), label_counts.setdefault("-", 0))


def get_attr_type(feature):
    return ATTR_TYPES[feature]


if __name__ == "__main__":

    # parse data
    data, meta = parse_arf(file("credit_train.arff"))

    # setup globals
    MAX_ELEMENTS_IN_NODE = int(argv[1])
    ATTR_TYPES = build_datatype_dict(meta)
    ATTR_NAMES = [x for x in meta]
    GAIN_THRESHOLD = 0

    # building the tree
    root = Node()
    build_tree(root, data)
    print_tree(root)

    # predicting on test set
    test_data, _ = parse_arf(file("credit_test.arff"))
    print "<Predictions for the Test Set Instances>"
    correct_count = 0
    wrong_count = 0
    for i, record in enumerate(test_data):
        actual = str(record[-1])
        predicted = predict(root, record)
        if actual == predicted:
            correct_count += 1
        print "%d: Actual: %s Predicted: %s" % (i + 1, actual, predicted)
    print "Number of correctly classified: %d Total number of test instances: %d" % (correct_count, len(test_data))
