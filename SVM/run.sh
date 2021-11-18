#!/bin/sh
echo -e "Part 2)\n"

echo -e "a) - first schedule"

echo -e "\n c = 100/873"
python3 ./svm.py 0.115
echo -e "\n c = 500/873"
python3 ./svm.py 0.573
echo -e "\n c = 700/873"
python3 ./svm.py 0.802

echo -e "\n"
echo -e "b) - second schedule\n"

echo -e "\n c = 100/873"
python3 ./svm.py 0.115 1
echo -e "\n c = 500/873"
python3 ./svm.py 0.573 1
echo -e "\n c = 700/873"
python3 ./svm.py 0.802 1

echo -e "\n\nPart 3)\n"
$SHELL