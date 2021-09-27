"""
Kidus Yohannes
Shandian Zhe
CS 5350 | Machine Learning
September 24, 2021

This code was written entirely by Kidus Yohannes
"""
import sys
import math
import statistics
import re
import os
import pdb # used for debuging


# Node used for representing a decision tree
class Node:
    # A node will either be a decision split (attribute) or a leaf (label)
    # The childs represent the connected nodes. Each decision (or attribute) value maps to another node,
    # which will either be another decision split or a leaf
    def __init__(self, attribute = None, label = None, childs = None):
        # This structure is necessary to prevent a previous bug where an empty constructor
        # wouldn't actually instaniate default values.
        if attribute is None:
            self.attribute = ""
        else:
            self.attribute = attribute

        if label is None:
            self.label = ""
        else:
            self.label = label

        if childs is None:
            self.childs = {}
        else:
            self.childs = childs

    # Overriding string method, for printing out the node tree structure
    def __str__(self, depth=0):
        ret = ""
        if self.attribute != "":
            ret += "[" + repr(self.attribute) + "]\n"
        else:
            ret += "= " + repr(self.label) + "\n"

        for key, value in self.childs.items():
            ret += "\t" * depth + repr(key)+ " --> "
            ret += value.__str__(depth+1)
        return ret
    

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


# Keeps track of the data from the given file.
# That means each example, the labels, attributes, and their values.
class Dataset:
    data = [] 
    labels = [] 
    attributes = [] 
    attribute_values = {}
    method_of_purity = ""
    max_depth = 0

    # Initialize using the description and training files
    def __init__(self, data_desc_file, train_file, method_of_purity, max_depth):
        # Get the labels, attributes, and their values from the data-desc file.
        #lines = []
        #with open(data_desc_file, 'r') as f:
        #    lines = f.readlines()
        #f.close()

        # Actually, just hard code it since parsing the file looks complicated
        self.labels = ["yes", "no"]
        self.attributes = ["age", "job", "marital", "education", "default", "balance",\
           "housing", "loan", "contact", "day", "month", "duration", "campaign",\
           "pdays", "previous", "poutcome"] 

        self.attribute_values["age"] = ["numeric"]
        self.attribute_values["job"] = ["admin.", "unknown", "unemployed", "management", "housemaid", "entrepreneur", "student", "blue-collar", "self-employed", "retired", "technician", "services"]
        self.attribute_values["marital"] = ["married", "divorced", "single"]
        self.attribute_values["education"] = ["unknown", "secondary", "primary", "tertiary"]
        self.attribute_values["default"] = ["yes", "no"]
        self.attribute_values["balance"] = ["numeric"]
        self.attribute_values["housing"] = ["yes", "no"]
        self.attribute_values["loan"] = ["yes", "no"]
        self.attribute_values["contact"] = ["unknown", "telephone", "cellular"]
        self.attribute_values["day"] = ["numeric"]
        self.attribute_values["month"] = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
        self.attribute_values["duration"] = ["numeric"]
        self.attribute_values["campaign"] = ["numeric"]
        self.attribute_values["pdays"] = ["numeric"]
        self.attribute_values["previous"] = ["numeric"]
        self.attribute_values["poutcome"] = ["unknown", "other", "failure", "success"]

        # Set method_of_purity to entropy (default), me, or gi
        self.method_of_purity = "entropy"
        if method_of_purity == "me":
            self.method_of_purity = "me"
        elif method_of_purity == "gi":
            self.method_of_purity = "gi"

        # Set max_depth
        self.max_depth = max_depth

        # Get the training data
        self.data = read_data(train_file, self.attributes)

        # Complete missing "unknown" value with most common value
        # First find most common values for each attribute
        most_common_values = {}
        for a in self.attributes:
            value_counts = {}
            for v in self.attribute_values[a]:
                value_counts[v] = len(list(filter(lambda x: x[a] == v, self.data)))
            value_counts.pop("unknown", None) # Removes unknown so it isn't the most common value
            most_common_values[a] = max(value_counts)

        # Then check each data example and replace any unknowns
        for d in self.data:
            for a in self.attributes:
                if d[a] == "unknown":
                    d[a] = most_common_values[a]
            

    # Calculates the entropy for a given data set
    def calc_entropy(self, data):
        # First calculate the proportion for each label value
        data_size = len(data)
        if data_size == 0:
            return 0
        label_proportions = []
        for l in self.labels:
            label_proportions.append(sum(x["label"] == l for x in data) / data_size)
    
        # Then sum the proportions times log (exact formula: -p * log_2(p))
        entropy = 0
        for p in label_proportions:
            if p != 0:
                entropy += -p * math.log(p, 2)

        return entropy

    # Calculates the majority error for a given data set
    def calc_me(self, data):
        # First calculate the proportion for each label value
        data_size = len(data)
        if data_size == 0:
            return 0

        label_proportions = []
        for l in self.labels:
            label_proportions.append(sum(x["label"] == l for x in data) / data_size)
        
        # Choose the smallest proportion error
        me = min(label_proportions)
        return me

    # Calculates the gini index for a given data set
    def calc_gi(self, data):
        # First calculate the proportion for each label value
        data_size = len(data)
        if data_size == 0:
            return 0
        label_proportions = []
        for l in self.labels:
            label_proportions.append(sum(x["label"] == l for x in data) / data_size)
        
        # Then sum the proportions squared, minus 1 (exact formula: 1 - sum (p^2))
        gi = 0
        for p in label_proportions:
            gi += p**2

        return 1 - gi

    # Chooses which purity equation to use based on the string given
    def calc_purity(self, data):
        purity = 0
        if self.method_of_purity == "entropy":
            purity = self.calc_entropy(data)
        elif self.method_of_purity == "me":
            purity = self.calc_me(data)
        elif self.method_of_purity == "gi":
            purity = self.calc_gi(data)
        return purity


    # Calculates the best attribute by choosing the one with the most information gain 
    def calc_best_attribute(self, data, attributes):
        # First calculate the current entropy
        current_purity = self.calc_purity(data)

        # Now for each attribute, we need to calculate the expected entropy and information gain
        data_size = len(data)
        attribute_info_gain = {}
        for a in attributes:
            a_expected_purity = []

            # If the values are numeric, find the median and turn the feature into a binary one
            if self.attribute_values[a] == ["numeric"]:
                numeric_values = []
                for d in data:
                    numeric_values.append(int(d[a]))
                numeric_median = statistics.median(numeric_values)
                
                # Less than median
                data_a_n = list(filter(lambda x: int(x[a]) < numeric_median, data))
                a_n_proportion = len(data_a_n) / data_size
                a_expected_purity.append(self.calc_purity(data_a_n) * a_n_proportion)
                
                #pdb.set_trace()

                # Greater than median
                data_a_n = list(filter(lambda x: int(x[a]) >= numeric_median, data))
                a_n_proportion = len(data_a_n) / data_size
                a_expected_purity.append(self.calc_purity(data_a_n) * a_n_proportion)

                # Sum purity and save it
                a_expected_purity = sum(a_expected_purity)
                attribute_info_gain[a] = current_purity - a_expected_purity
            else:
                for v in self.attribute_values[a]:
                    data_a_v = list(filter(lambda x: x[a] == v, data))
                    a_v_proportion = len(data_a_v) / data_size
                    a_expected_purity.append(self.calc_purity(data_a_v) * a_v_proportion)
                a_expected_purity = sum(a_expected_purity)
                attribute_info_gain[a] = current_purity - a_expected_purity
    
        # The best attribute to split on is the one that has the most information gain
        best_attribute = max(attribute_info_gain, key=attribute_info_gain.get)
        return best_attribute


    # ID3 algorithm, recursively calls itself
    def id3(self, data, attributes, depth = 0):
        # If all examples have the same label, return a leaf node with that label
        data_labels = []
        for d in data:
            data_labels.append(d["label"])
        if data_labels.count(data_labels[0]) == len(data_labels):
            return Node("", data_labels[0], {})

        # If attributes is empty, return a leaf node with the most common label
        data_labels_count = {}
        if len(attributes) == 0:
            for l in self.labels:
                data_labels_count[l] = data_labels.count(l)
        
            most_common_label = max(data_labels_count, key=data_labels_count.get)
            return Node("", most_common_label, {})

        # Otherwise
        # Create a root node
        root = Node()
        
        # Stop building the tree when we reach the max_depth
        if depth == self.max_depth:
            # Just return the most common label
            data_labels_count = {}
            for l in self.labels:
                data_labels_count[l] = data_labels.count(l)
        
            most_common_label = max(data_labels_count, key=data_labels_count.get)
            root.label = most_common_label
            return root

        # Get best attribute to split on
        best_attribute = self.calc_best_attribute(data, attributes)
        root.attribute = best_attribute
        
        # Do things slighlty differently for numeric values
        if self.attribute_values[best_attribute] == ["numeric"]:
            numeric_values = []
            for d in data:
                numeric_values.append(int(d[best_attribute]))
            numeric_median = int(statistics.median(numeric_values))
                
            # Get the subset of examples where the best attribute is less than the median
            data_ba_n= list(filter(lambda x: int(x[best_attribute]) < numeric_median, data))
            
            # If the subset is empty, add a leaf node with the most common label from the whole data set
            if len(data_ba_n) == 0:
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
                root.childs["< " + str(numeric_median)] = self.id3(data_ba_n, attributes_, depth + 1)
                #pdb.set_trace()

            # Get the subset of examples where the best attribute is greater than the median
            data_ba_n= list(filter(lambda x: int(x[best_attribute]) >= numeric_median, data))
            
            # If the subset is empty, add a leaf node with the most common label from the whole data set
            if len(data_ba_n) == 0:
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
                root.childs[">= " + str(numeric_median)] = self.id3(data_ba_n, attributes_, depth + 1)
        else:
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
                    root.childs[v] = self.id3(data_ba_v, attributes_, depth + 1)
                    #pdb.set_trace()
        
        # Finally return root node
        return root

def main():
    # Read the chosen method for purity (entropy, me, or gi)
    method_of_purity = "entropy" if len(sys.argv) == 1 else sys.argv[1]
    max_depth = sys.maxsize if len(sys.argv) <= 2 else int(sys.argv[2])
    
    # Set up the file names and dataset variable
    data_desc_file = os.path.join("bank", "data-desc.txt")
    train_file = os.path.join("bank", "train.csv")
    test_file = os.path.join("bank", "test.csv")
    dataset = Dataset(data_desc_file, train_file, method_of_purity, max_depth)
    
    # Use recursive ID3 algorithm to create a decision tree for the dataset
    decision_tree = dataset.id3(dataset.data, dataset.attributes)
    
    # Print the resulting tree
    print(decision_tree)

    # Now test the decision tree on the test file
    # Loop through each example in the test data and keep track of the errors
    test_data = read_data(test_file, dataset.attributes)
    prediction_errors = 0
    for d in test_data: 
        # Go through our decision tree and find the label prediction
        label_prediction = ""
        current_node = decision_tree
        while True:
            decision = current_node.attribute
            if decision == "":
                label_prediction = current_node.label
                break
            decision_value = d[decision]
            # If the value is numeric, compare to the median to determine which branch to take
            if dataset.attribute_values[decision] == ["numeric"]:
                #pdb.set_trace()
                numeric_median = int(list(current_node.childs.keys())[0][2:])
                if int(decision_value) < numeric_median:
                    current_node = current_node.childs["< " + str(numeric_median)]
                else:
                    current_node = current_node.childs[">= " + str(numeric_median)]
            else:
                current_node = current_node.childs[decision_value]
        
        # Keep track of incorrect predictions
        if d["label"] != label_prediction:
            prediction_errors += 1
    error_percentage = prediction_errors / len(test_data)

    # Print results
    print("Purity: " + method_of_purity + "\t Max Depth: " + str(max_depth))
    print("Error percentage: " + str(error_percentage))


if __name__ == "__main__":
    main()
