# [Machine Learning Notebook](https://calvinfeng.gitbook.io/machine-learning-notebook/)

## Introduction

This is my personal notebook for documenting knowledge I picked up as I progress through my career
in machine learning. It helps me reinforcing my theoretical understanding of the problems and
solutions I have seen.

## Python 2 vs Python 3

I wrote majority of the content in Python 2.7 in 2018. Now it's 2022 April, Python 2 has been long
deprecated, I am switching to Python 3.8 with TensorFlow 2.8.

My current system setup

- Ubuntu 20.04
- Tensorflow 2.8.*
- Python 3.8.*
- CUDA 11.2
- cuDNN 8.4
- Matplotlib 3.5.*

Some older code will be running on

- Tensorflow 1.15
- Python 2.7.*

## Table of Contents

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
* Machine Learning in Production

## Export Notebook

### Jupyter Convert

If my notebook does not contain any `matplotlib.pyplot` then I can export it as simple text.

```bash
jupyter nbconvert --to markdown loss_function_overview.ipynb --stdout
```

Otherwise, I'd need to export differently.

```bash
jupyter nbconvert --to markdown loss_function_overview.ipynb
```

### Latex

Jupyter notebook uses single dollar sign for inline equations but GitBook uses double dollar sign
for inline equations. I need a RegExp that capture and convert.

`?` means once or none.

```regexp
\$.?\$
```

`+` means one or more.

```regexp
\$.+\$
```

The following will capture all `$<some text>$`.

```regexp
^\$.+\$$
```
