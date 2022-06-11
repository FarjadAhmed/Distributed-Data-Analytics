#!/usr/bin/python3
# -*- coding: utf-8 -*-
import sys

mydict1 = {}
tuplist = []

for line in sys.stdin:
    origin, arr_delay = line.strip().split('\t')
    arr_delay = float(arr_delay)

    if origin in mydict1:
        mydict1[origin].append(arr_delay)
    else:
        ddelay_list = []
        ddelay_list.append(arr_delay)
        mydict1[origin] = ddelay_list
for key in mydict1:
    mydict1[key] = round(sum(mydict1[key])/len(mydict1[key]), 2)
    tuplist.append(tuple((key, mydict1[key])))
    tuplist.sort(key=lambda y: y[1], reverse=True)

for i in range(10):
    print(tuplist[i][0], tuplist[i][1])

    

    
