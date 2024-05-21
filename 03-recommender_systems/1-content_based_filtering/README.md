# Programming Assignment: Deep Learning for Content-Based Filtering

> Implement a content-based collaborative filtering recommender system for movies. This lab will use neural networks to generate 
the user and movie vectors.

## Packages
We will use familiar packages, NumPy, TensorFlow and helpful routines from [scikit-learn](https://scikit-learn.org/stable/). 
We will also use [tabulate](https://pypi.org/project/tabulate/) to neatly print tables and [Pandas](https://pandas.pydata.org/) 
to organize tabular data.
```
import numpy as np
import numpy.ma as ma
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import tabulate
from recsysNN_utils import *
pd.set_option("display.precision", 1)
```

## Content-based filtering with a neural network

In the collaborative filtering lab, you generated two vectors, a user vector and an item/movie vector whose dot product would 
predict a rating. The vectors were derived solely from the ratings.   

Content-based filtering also generates a user and movie feature vector but recognizes there may be other information available 
about the user and/or movie that may improve the prediction. The additional information is provided to a neural network which then generates the user and movie vector.

## Neural Network for content-based filtering
Now, let's construct a neural network. It will have two networks that are combined by a dot product. You will construct the two networks. In this example, 
they will be identical. Note that these networks do not need to be the same. If the user content was substantially larger than the movie content, you might 
elect to increase the complexity of the user network relative to the movie network. In this case, the content is similar, so the networks are the same.

<a name="ex01"></a>
### Exercise 1

- Use a Keras sequential model
    - The first layer is a dense layer with 256 units and a relu activation.
    - The second layer is a dense layer with 128 units and a relu activation.
    - The third layer is a dense layer with `num_outputs` units and a linear or no activation.   
    
The remainder of the network will be provided. The provided code does not use the Keras sequential model but instead uses the Keras [functional api](https://keras.io/guides/functional_api/). This format allows for more flexibility in how components are interconnected.

<a name="ex02"></a>
### Exercise 2

Write a function to compute the square distance.