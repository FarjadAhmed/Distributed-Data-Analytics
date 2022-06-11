#!/usr/bin/python3
# -*- coding: utf-8 -*-
import sys

last_movie = None
mydict1 = {}
tuplist = []

for line in sys.stdin:
    movie, rating = line.strip().split('\t')
    rating = float(rating)
    if movie in mydict1:
        mydict1[movie].append(rating)
    else:
        rating_list = []
        rating_list.append(rating)
        mydict1[movie] = rating_list
for key in mydict1:
    mydict1[key] = round(sum(mydict1[key])/len(mydict1[key]), 2)
    tuplist.append(tuple((key, mydict1[key])))
tuplist.sort(key=lambda y: y[1], reverse=True)

print(tuplist[0][0], '\t', tuplist[0][1])

    

    
