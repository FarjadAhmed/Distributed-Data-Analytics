#!/usr/bin/python3

import sys
import re
next(sys.stdin)  
title = ''
for line in sys.stdin:
    data = line.strip().split(',')
    length = len(data)
    if length < 5:
        next(sys.stdin)
    elif length == 5:
        print(data[0],'\t',data[3])
    elif length >5:
        for i in range((length-5)+1):
            title = title+','+data[i]
        print(title,'\t',data[-2])
        title = ''

