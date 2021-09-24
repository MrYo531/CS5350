"""
Kidus Yohannes
Shandian Zhe
CS 5350 | Machine Learning
September 24, 2021

This code was written entirely by Kidus Yohannes
"""
import math
import re
from pprint import pprint
import pdb


# Node used for representing a decision tree
class Node:
    # A node will either be a decision split (attribute) or a leaf (label)
    # The childs represent the connected nodes. Each decision (or attribute) value maps to another node,
    # which will either be another decision split or a leaf
    def __init__(self, attribute = "", label = "", childs = {}):
        self.attribute = attribute
        self.label = label
        self.childs = childs
    

# From a given data set, formats and returns it as a list of dictionaries
def read_data(CSV_file, attributes):
    data = []
    with open(CSV_file, 'r') as f:
        for line in f:
            terms = line.strip().split(',')
            values = {}
            i = 0
            for a in attributes:
                values[a] = terms[i]
                i += 1

            values["label"] = terms[i]
            data.append(values)
    f.close()
    return data


 
# Recursively creates a decision tree
def dt_recursive(data, label_values, attributes, attribute_values):
    # Calculate the current entropy
    current_entropy = calc_entropy(data, label_values)

    # Now for each attribute, we need to calculate the expected entropy and information gain
    data_size = len(data)
    attribute_info_gain = {}
    for a in attributes:
        a_expected_entropy = []
        for v in attribute_values[a]:
            data_a_v = list(filter(lambda x: x[a] == v, data))
            a_v_proportion = len(data_a_v) / data_size
            #if len(data_a_v) != 0:
            a_expected_entropy.append(calc_entropy(data_a_v, label_values) * a_v_proportion)
        a_expected_entropy = sum(a_expected_entropy)
        attribute_info_gain[a] = current_entropy - a_expected_entropy

    # Choose the best attribute to split on and set up the root node with it
    root = Node()
    best_attribute = max(attribute_info_gain, key=attribute_info_gain.get)
    root.attribute = best_attribute
    
    # Check each branch, split recursively or stop if all the leaves have the same label
    for v in attribute_values[best_attribute]:
        # Get the label values
        leaf_label_values = []
        data_ba_v= list(filter(lambda x: x[best_attribute] == v, data))
        for d in data_ba_v:
            leaf_label_values.append(d["label"])

        # If empty, add a leaf node with the most common label value
        if len(data_ba_v) == 0:
            label_count = {}
            for l in label_values:
                label_count[l] = len(list(filter(lambda x: x["label"] == l, data)))
            print(label_count)
            most_common_label = max(label_count, key=label_count.get)
            root.childs[v] = most_common_label


        # All the leaves have the same label
        #len(leaf_label_values) != 0 and
        if leaf_label_values.count(leaf_label_values[0]) == len(leaf_label_values):
            root.childs[v] = leaf_label_values[0]
        else:
            attributes_ = attributes.copy()
            attributes_.remove(best_attribute)
            
            if len(attributes_) == 1:
                pdb.set_trace()
            root.childs[v] = dt_recursive(data_ba_v, label_values, attributes_, attribute_values);

    return root


# Keeps track of the data from the given file.
# That means each example, the labels, attributes, and their values.
class Dataset:
    data = [] 
    labels = [] 
    attributes = [] 
    attribute_values = {}

    # Initialize using the description and training files
    def __init__(self, data_desc_file, train_file):
        # Get the labels, attributes, and their values from the data-desc file.
        lines = []
        with open(data_desc_file, 'r') as f:
            lines = f.readlines()
        f.close()

        self.labels = lines[2].strip().replace(" ", "").split(',')
        self.attributes = lines[len(lines) - 1].strip().split(',')[:-1]

        for a in self.attributes:
            for l in lines:
                if (a in l):
                    values = re.split(',|:|\.', l.strip().replace(" ", ""))[1:-1]
                    self.attribute_values[a] = values
                    break

        # Get the training data
        self.data = read_data(train_file, self.attributes)

    # Calculates the entropy for a given data set
    def calc_entropy(self, data):
        # First calculate the proportion for each label value
        data_size = len(data)
        label_proportions = []
        for l in self.labels:
            label_proportions.append(sum(x["label"] == l for x in data) / data_size)
    
        # Then sum the proportions times log (exact formula: -p * log_2(p))
        entropy = 0
        for p in label_proportions:
            if p != 0:
                entropy += -p * math.log(p, 2)

        return entropy;

    # Calculates the best attribute by choosing the one with the most information gain 
    def calc_best_attribute(self):
        # First calculate the current entropy
        current_entropy = self.calc_entropy(self.data)
    
        # Now for each attribute, we need to calculate the expected entropy and information gain
        data_size = len(self.data)
        attribute_info_gain = {}
        for a in self.attributes:
            a_expected_entropy = []
            for v in self.attribute_values[a]:
                data_a_v = list(filter(lambda x: x[a] == v, self.data))
                a_v_proportion = len(data_a_v) / data_size
                a_expected_entropy.append(self.calc_entropy(data_a_v) * a_v_proportion)
            a_expected_entropy = sum(a_expected_entropy)
            attribute_info_gain[a] = current_entropy - a_expected_entropy
    
        # The best attribute to split on is the one that has the most information gain
        best_attribute = max(attribute_info_gain, key=attribute_info_gain.get)
        return best_attribute



    # ID3 algorithm, recursively calls itself
    def id3(self, data, attributes):
        # If all examples have the same label, return a leaf node with that label
        data_labels = []
        for d in data:
            data_labels.append(d["label"])
        if data_labels.count(data_labels[0]) == len(data_labels):
            return Node("", data_labels[0], {});

        # If attributes is empty, return a leaf node with the most common label
        data_labels_count = {}
        if len(attributes) == 0:
            for l in self.labels:
                data_labels_count[l] = data_labels.count(l)
        
            most_common_label = max(data_labels_count, key=data_labels_count.get)
            return Node("", most_common_label, {})

        # Otherwise
        # Create a root node
        node = Node()
        
        # Get best attribute to split on
        best_attribute = self.calc_best_attribute()
        node.attribute = best_attribute
        
        # For each possible value for the best attribute
        for v in self.attribute_values[best_attribute]:
            # Get the subset of examples where the best attribute equals the value
            data_ba_v= list(filter(lambda x: x[best_attribute] == v, data))
            
            # If the subset is empty, add a leaf node with the most common label from the whole data set
            if len(data_ba_v) == 0:
                data_labels = []
                for d in data:
                    data_labels.append(d["label"])

                data_labels_count = {}
                for l in self.labels:
                    data_labels_count[l] = data_labels.count(l)
 
                most_common_label = max(data_labels_count, key=data_labels_count.get)
                return Node("", most_common_label, {})
            # Else add the subtree by recursively calling the id3 algorithm
            else:
                attributes_ = attributes.copy()
                attributes_.remove(best_attribute)
                node.childs[v] = self.id3(data_ba_v, attributes_)
        
        # Finally return root node
        return node;   


def main():
    data_desc_file = "car\\data-desc.txt"
    train_file = "car\\train.csv"
    dataset = Dataset(data_desc_file, train_file)
    
    #dataset.print_data()
    
    #data_safety_low = list(filter(lambda x: x["safety"] == "low", data))

    # Use recursive ID3 algorithm to create decision tree for the data
    root = dataset.id3(dataset.data, dataset.attributes)

    pprint(vars(root))

    # Calculate the current entropy
    #current_entropy = calc_entropy(data, label_values)
    #
    ## Now for each attribute, we need to calculate the expected entropy and information gain
    #data_size = len(data)
    #attribute_info_gain = {}
    #for a in attributes:
    #    a_expected_entropy = []
    #    for v in attribute_values[a]:
    #        data_a_v = list(filter(lambda x: x[a] == v, data))
    #        a_v_proportion = len(data_a_v) / data_size
    #        a_expected_entropy.append(calc_entropy(data_a_v, label_values) * a_v_proportion)
    #    a_expected_entropy = sum(a_expected_entropy)
    #    attribute_info_gain[a] = current_entropy - a_expected_entropy
    #
    ## Choose the best attribute to split on and set up the root node with it
    #root = Node()
    #best_attribute = max(attribute_info_gain, key=attribute_info_gain.get)
    #root.attribute = best_attribute
    #
    ## Check each branch, split recursively or stop if all the leaves have the same label
    #for v in attribute_values[best_attribute]:
    #    # Get the label values
    #    leaf_label_values = []
    #    data_ba_v= list(filter(lambda x: x[best_attribute] == v, data))
    #    for d in data_ba_v:
    #        leaf_label_values.append(d["label"])
    #
    #    # If empty, add a leaf node with the most common label value
    #    if len(data_ba_v) == 0:
    #        label_count = {}
    #        for l in label_values:
    #            label_count[l] = len(list(filter(lambda x: x["label"] == l, data)))
    #        print(label_count)
    #        most_common_label = max(label_count, key=label_count.get)
    #        root.childs[v] = most_common_label
    #
    #    # All the leaves have the same label
    #    if(leaf_label_values.count(leaf_label_values[0]) == len(leaf_label_values)):
    #        root.childs[v] = leaf_label_values[0];
    #    else:
    #        attributes_ = attributes.copy()
    #        attributes_.remove(best_attribute)
    #        #pdb.set_trace()
    #        root.childs[v] = dt_recursive(data_ba_v, label_values, attributes_, attribute_values);
    #
    #pprint(vars(root))


if __name__ == "__main__":
    main()