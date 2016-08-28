---
layout: post
title: Dimension reduction with random projections, part 1
---

Lets start with a mental experiment.
Imagine that we are working with brain data.
Our imaginary dataset will have $$ n $$ brain images, each one composed of $$ p
$$ voxels -- a voxel is an unit of 3D images, like pixels are to 2D images
--r. Usually, the number $$ p $$ of voxels is arround hundreds of thousands ~ $$ 100.000 $$. 

Now, suppose that these brain images were taken from a medical study,
because we are willing to do something meaningful with our data.
As we are now working with brain data, it'll be nice to see if a subject has a disease or not.
If the take a machine learning approach, we'll have to gather a bunch of brains
to be able to train a classifier -- e.g. a Suppor Vector Machine or Logistic
Regression --.
In this setting the voxels are the features.

Ok, so now we have bunch of brains and even more voxels to train a classifier. 
But, we have at most some hundreds of subjects while the number of voxels remains hundreds of thousands.
This is what people call an **High dimensional** setting.

High dimensional data poses some challenges, both form the analytical and
computational point of view.
From a computational perpective, somethimes we can't fit the data in memory,
and when we can, the classifation algorithm is going to take a huge amount of
time to give us a solution, and we are going to need several solutions or models to validate the result.

*One possible solution:* To reduce the number $$ p $$ of voxels, this is
going to alleviate cache problems and it'll help us to find models faster.
There are several methods to perform dimension reduction like Matrix
Decomposition --e.g. Principarl Component Analysis (PCA), Independent
Component Analysis (ICA), Dictionary Learning, etc. --, Feature Clustering,
Screening, Random Projections, among others.

Our aim here is to provide an introduction to random projections as a dimension
reduction technique, but hopefully we'll be able to see them working together
with other methods like the PCA.


## Dimension reduction ##

Basically, the classifiers use the information of how similar two brains are to decide whether the subject is or isn't sick.
In other words, healthy brains are similar among them and it is the same for
the sick ones.

There are many ways to measure similarity, like correlation or Euclidean distance for instance, each one with its own advantages and drawbacks.
Here, lets focus on latter: the Euclidean norm, $$ \|x\|_2^2 = \sum_i {x_i ^ 2}$$.





<div class="col-sm-12">
<img src=" {{" /img/dim_reduction.png " | prepend: site.baseurl }}" align="middle" altt="Dimension reduction">
</div>



$$
\begin{equation}
\|x - x^\prime \|_2^2 \leq \|f(x) - f(x^\prime)\|_2^2 \leq \|x - x^\prime\|_2^2
\label{jl}
\end{equation}
$$


This is known as the Johnson-Lindenstrauss lemma \ref{jl}

$$\mathbf{\Phi} \in \mathbb{R}^{k \times p}$$


```python
import numpy as np

x = np.zeros(0, 0)
```


