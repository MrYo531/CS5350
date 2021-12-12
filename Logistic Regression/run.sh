#!/bin/sh
echo -e "Part a) - MAP estimation\n"

python3 ./lr.py map

echo -e "\n"
echo -e "Part b) - ML estimation\n"

python3 ./lr.py ml


$SHELL