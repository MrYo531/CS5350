"""
Kidus Yohannes
Shandian Zhe
CS 5350 | Machine Learning
September 24, 2021

This code was written entirely by Kidus Yohannes
"""

CSVfile = "car\\train.csv"

with open(CSVfile, 'r') as f:
    for line in f:
        terms = line.strip().split(',')
        print(terms)