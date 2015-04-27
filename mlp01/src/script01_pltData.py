#!/usr/bin/env python
# coding: utf-8
#
# Author: Peinan ZHANG
# Created at: 2015-04-27

import matplotlib.pyplot as plt
import scipy as sp

data = sp.genfromtxt('data/web_traffic.tsv', delimiter='\t')
x = data[:, 0]
y = data[:, 1]

plt.scatter(x,y)
plt.title("Web traffic over the last month")
plt.xlabel("Time")
plt.ylabel("Hits/hour")
plt.xticks([w*7*24 for w in range(10)], ['week %i'%w for w in range(10)])
plt.autoscale(tight=True)
plt.grid()
plt.show()
