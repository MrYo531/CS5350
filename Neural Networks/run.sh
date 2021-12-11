#!/bin/sh
echo -e "Part a) - back-propagation\n"

python3 ./nn.py bp

echo -e "\n"
echo -e "Part b) - SGD algorithm with different widths\n"

python3 ./nn.py sgd

echo -e "\n"
echo -e "Part c) - weights init at 0\n"

python3 ./nn.py sgd_0

$SHELL