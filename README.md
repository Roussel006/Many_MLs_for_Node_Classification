# Comparing and Combining Different Machine Learning Models for Node Classification Tasks

How to predict a value from examples when we do not know the relationships between our inputs (i.e., the examples) and outputs? This question drives Machine Learning and is also relevant to some other fields (such as control and optimization). In this age of AIs and LLMs, there are more choices for a model than the fingers on our hands. Some datasets may contain different types of information that may be suitable for different types of models. For example, datasets like CiteSeer, Cora, and Pubmed contain both the text and the graph data. We may build pure text-based models, or we may incorporate the relationships represented in the graph structure. For some of the datasets, we also have image data that can be incorporated. In short, different models vary in their architectures, suitability to data modalities, strengths, and weaknesses. Among all these choices and their combinations, the choice can be difficult. 

What may help us get started is looking at the same task being done in different ways. Here, we build a host of models for the same classification task -- Node classification of the CiteSeer Dataset -- and explore what happens when we combine different models and modalities of information for the class prediction task. First, we have the "shallow" ML approaches, such as Random Forest, Naive Bayes, XGBoost to name a few. Second, we have the "Deep" Learning (DL) models that enable us to use more complex relationships in the data. Third, we have the Graph-based ML (Graph ML) approaches that incorporate link information within the  into the mix. Fourth, we also have Bayesian Machine Learning models. Finally, we can combine these models to see how well our "army" or ensemble of models does together.


<!-- In short, the choice is not uniquand requires trial-and-error with different models to find the best one from the pool. In this work, we run a set of experiments for the same classification task -- Node classification of the CiteSeer Dataset -- and compare their performance. We also see what happens if we stack or combine different models together. -->

## List of Experiments

**Section1**: "Shallow" ML algorithms

In this section, we will build eight "shallow" ML classifiers. We will learn how to implement each of them using Scikit-Learn and save the models for later comparisons with other models. The models we will develop are:

    1a. Naive Bayes
    1b. XGBoost
    1c. Decision Trees
    1d. Random Forest
    1e. Gradient Boosting
    1f. CatBoost
    1g. LightGBM
    1h. Support Vector Machine (SVM) Classifiers

**Section 2**: Deep Learning Models 

In this section, we build five DL models, using **Tensorflow**. Later, we will see how to build the same models to PyTorch and learn how to translate models between Tensorflow and PyTorch.

    2a. A simple dense model: with only pooling layer between the input and the output layers.
    2b. A deeper model with more layers
    2c. An Recurrent Neural Network (RNN) with Long Short Term Memory (LSTM)
    2d. An RNN with Gated Recurrent Unit (GRU)
    2e. A Convolutional Neural Network (CNN)

**Section 3**: Machine Learning with Graphs 

In this section, we will build several Graph-based MLs. There are several types of ML with Graphs -- Feature Engineering (for Graph Characteristics as features), Graph Representation Learning, Graph Neural Networks, and more. Here, we will implement and compare some of them.

    3a. A Graph Convolutional Network (GCN)
    3b. A Graph Attention Network (GAT)

A note on resources: 
    
    1. A great open-source resource for ML with Graphs is the the CS224W course by Jure Lescovec at Stanford. The lecture slides, notebooks, and problem sets for several years are provided in entirety (Thanks a lot, Dr. Lescovec!), such as the Fall 2023 resource can be found here: https://snap.stanford.edu/class/cs224w-2023/. Lecture videos for particular years can be found on YouTube -- Isn't that just awesome? Here is the lecture series for 2021: https://youtu.be/JAB_plj2rbA?si=ujCiFl6HBFSqlGVf
    2. The [Graph Neural Network Course](https://github.com/mlabonne/Graph-Neural-Network-Course) by [@maximelabonne](https://twitter.com/maximelabonne). Some of the codes here will be developed based on Chapter 2 of the course and this companion article: https://mlabonne.github.io/blog/gat/
    3. A quick introduction: [Hugging Face blog](https://huggingface.co/blog/intro-graphml).

**Section 4**: Bayesian Machine Learning

In this section, we will use **PyMC3** to build some Bayesian ML models. In this networks, we will assume some prior distributions over the model parameters and after training, we will find the posterior distributions of the parameters. We will MCMC 

**Section 5**: Ensembling or Stacking different models

Here, we will try different combinations of all the models we have built, to see which ones are the best ones.



## Goals:
    1. Compare the performance of different ML models and their combinations for class prediction.
    2. Compare different implementations (e.g., Tensorflow and PyTorch)

<!-- ## More options:

PyMC3 models -- Bayesian Learning models. Gaussian processes, 
Unsupervised methods (e.g., Spectral Clustering) -->
