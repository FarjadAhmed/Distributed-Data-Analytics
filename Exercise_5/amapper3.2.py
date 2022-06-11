#!/usr/bin/python3


import sys
import re
next(sys.stdin)  

for line in sys.stdin:
    data = line.strip().split(',')
    print(data[-1],'\t',data[-2])


