#!/usr/bin/python3

import sys
import re

next(sys.stdin)  

for line in sys.stdin:
    data = line.strip().split(',')
    print(data[15],'\t',data[45])

