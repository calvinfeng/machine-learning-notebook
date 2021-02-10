# [Machine Learning Notebook](https://calvinfeng.gitbook.io/machine-learning-notebook/)

## Introduction

This is my personal machine learning notebook, used primarily for references and sharing what I
learned recently.

The source of the content primarily comes from courses I took from Stanford, i.e. CS229, CS231n and
CS224n and many other research papers, textbooks and online tutorials. There are also notes I took
from my professional work and personal projects using machine learning.

## Python 2 vs Python 3

I wrote majority of the content in Python 2.7 in 2018. Now Python 2.7 is fully deprecated, I am
switching to Python 3.8.

My dependency will be upgraded to

- Tensorflow 2.4.1
- Keras 2.4.3

Some of my old Tensorflow code probably won't work anymore.

## Table of Contents

The list is not sorted in any order.

* Clustering
* Simple Neural Networks
* Convolutional Neural Networks
* Generative Adversial Networks
* Recurrent Neural Networks
* Random Forest
* Reinforcement Learning
* Natural Language Processing
* Naive Bayesian Networks
* Recommender System
* Transferred Learning
* Machine in Learning in Production

The list is certainly expanding as I take new classes and learn new things from work.

## Export Notebook

### Jupyter Convert

If my notebook does not contain any `matplotlib.pyplot` then I can export it as simple text.

```bash
jupyter nbconvert --to markdown loss_function_overview.ipynb --stdout
```

Otherwise, I'd need to export differently.

### Latex

Jupyter notebook uses single dollar sign for inline equations but GitBook uses double dollar sign
for inline equations. I need a RegExp that capture and convert.

```text
\$.*?\$
```
