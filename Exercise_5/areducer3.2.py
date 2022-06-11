#!/usr/bin/python3
# -*- coding: utf-8 -*-
import sys

last_user = None
mydict1 = {}
tuplist = []
shortlisted_users = {}

for line in sys.stdin:
    user, rating = line.strip().split('\t')
    rating = float(rating)
    user = str(user)
    if user in mydict1:
        mydict1[user].append(rating)
    else:
        rating_list = []
        rating_list.append(rating)
        mydict1[user] = rating_list
for key in mydict1:
    getlist = mydict1[key]
    if len(getlist) > 40:
        shortlisted_users[key] = round(sum(mydict1[key])/len(mydict1[key]), 2)
        tuplist.append(tuple((key, shortlisted_users[key])))
tuplist.sort(key=lambda y: y[1], reverse=False)

print(tuplist[0][0], '\t', tuplist[0][1])

    

    

