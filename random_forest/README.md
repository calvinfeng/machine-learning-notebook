This is the shitty one

https://www.analyticsvidhya.com/blog/2016/04/complete-tutorial-tree-based-modeling-scratch-in-python/

The better one is on Google channel.


## Gini Impurity
We can quantify the uncertainty as a single node using a metric called *Gini Impurity*.

## Information Gain
We can quantify how much does a question reduce that uncertainty using a concept called *Information Gain*.

## Question To Ask
What type of question can we ask at each node? Notice that each node takes a list of rows as input.
We will iterate over every value for every feature that appears in those rows. Each of these become 
a candidate for threshold we can use to partition the data.