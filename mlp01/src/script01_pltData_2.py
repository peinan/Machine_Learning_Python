#!/usr/bin/env python
# coding: utf-8
#
# Author: Peinan ZHANG
# Created at: 2015-04-27

import matplotlib.pyplot as plt
import scipy as sp

def error(f, x, y):
  return sp.sum((f(x) - y) ** 2)

data = sp.genfromtxt('data/web_traffic.tsv', delimiter='\t')
x = data[:, 0]
y = data[:, 1]

fp1, residuals, rank, sv, rcond = sp.polyfit(x, y, 1, full=True)
f1 = sp.poly1d(fp1)
fx = sp.linspace(0,x[-1], 1000)
plt.plot(fx, f1(fx), linewidth=4)
plt.legend(["d=%i" % f1.order], loc="upper left")

plt.scatter(x,y)
plt.title("Web traffic over the last month")
plt.xlabel("Time")
plt.ylabel("Hits/hour")
plt.xticks([w*7*24 for w in range(10)], ['week %i'%w for w in range(10)])
plt.autoscale(tight=True)
plt.grid()
plt.show()
