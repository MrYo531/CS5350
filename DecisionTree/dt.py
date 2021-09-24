"""
Kidus Yohannes
Shandian Zhe
CS 5350 | Machine Learning
September 24, 2021

This code was written entirely by Kidus Yohannes
"""
import math
import re

# From a given data set, formats and returns it as a list of dictionaries
def read_data(CSVfile, attributes):
    data = []
    with open(CSVfile, 'r') as f:
        for line in f:
            terms = line.strip().split(',')
            values = {}
            i = 0
            for a in attributes:
                values[a] = terms[i]
                i += 1

            values["label"] = terms[6]
            data.append(values)
    f.close()
    return data

# Calculate the entropy for a given data set
def calc_entropy(data, label_values):
    # First calculate the proportion for each label value
    data_size = len(data)
    label_proportions = []
    for v in label_values:
        label_proportions.append(sum(x["label"] == v for x in data) / data_size)
    
    # Then sum the proportions times log (exact formula: -p * log_2(p))
    entropy = 0
    for p in label_proportions:
        if p != 0:
            entropy += -p * math.log(p, 2)

    return entropy;
 
def main():
    # Get the label values, attributes, and their values from the data-desc file.
    data_desc_file = "car\\data-desc.txt"
    lines = []
    with open(data_desc_file, 'r') as f:
        lines = f.readlines()
    f.close()

    label_values = lines[2].strip().replace(" ", "").split(',')
    attributes = lines[len(lines) - 1].strip().split(',')[:-1]
    attribute_values = {}
    for a in attributes:
       for l in lines:
           if (a in l):
               values = re.split(',|:|\.', l.strip().replace(" ", ""))[1:-1]
               attribute_values[a] = values
               break
    
    # Get the training data
    train_file = "car\\train.csv"
    data = read_data(train_file, attributes)

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
            a_expected_entropy.append(calc_entropy(data_a_v, label_values) * a_v_proportion)
        a_expected_entropy = sum(a_expected_entropy)
        attribute_info_gain[a] = current_entropy - a_expected_entropy

    # Split on the maximum info gain and recursively repeat untill all the leaves have the same label
    best_attribute = max(attribute_info_gain, key=attribute_info_gain.get)

    print(attribute_info_gain);
    print(best_attribute)

if __name__ == "__main__":
    main()