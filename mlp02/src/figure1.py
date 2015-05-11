#!/usr/bin/env python
# coding: utf-8
#
# Author: Peinan ZHANG
# Created at: 2015-05-10

from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
import numpy as np

data = load_iris()

feats = data['data']
feat_names = data['feature_names']
target = data['target']
target_names = data['target_names']
labels = target_names[target]

pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
for i, (p0, p1) in enumerate(pairs):
  plt.subplot(2, 3, i + 1)
  for t, marker, c in zip(range(3), ">ox", "rgb"):
    plt.scatter(feats[target == t, p0],
                feats[target == t, p1],
                marker=marker,
                c=c)
  plt.xlabel(feat_names[p0])
  plt.ylabel(feat_names[p1])
  plt.xticks([])
  plt.yticks([])

plt.savefig('../img/figure2-1.png')
