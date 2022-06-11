#!/usr/bin/python3
# -*- coding: utf-8 -*-
import sys

mydict1 = {}
for line in sys.stdin:
    origin, dep_delay = line.strip().split('\t')
    dep_delay = float(dep_delay)
    #print(origin, dep_delay)

    if origin in mydict1:
        mydict1[origin].append(dep_delay)
    else:
        ddelay_list = [dep_delay]
        mydict1[origin] = ddelay_list
for key in mydict1:
    print( key,' min:', min(mydict1[key]), 'max:', max(mydict1[key]), 'avg:', round(sum(mydict1[key])/len(mydict1[key]), 2))

    
