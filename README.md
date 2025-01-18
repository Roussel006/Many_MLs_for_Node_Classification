# Comparing and Combining Different Machine Learning Models for Node Classification Tasks

How to predict a value from examples when we do not know the relationships between our inputs (i.e., the examples) and outputs? This question drives Machine Learning and is also relevant to some other fields (such as control and optimization). In this age of AIs and LLMs, there are more choices for a model than the fingers on our hands. Some datasets may contain different types of information that may be suitable for different types of models. For example, datasets like CiteSeer, Cora, and Pubmed contain both the text and the graph data. We may build pure text-based models, or we may incorporate the relationships represented in the graph structure. For some of the datasets, we also have image data that can be incorporated. In short, different models vary in their architectures, suitability to data modalities, strengths, and weaknesses. Among all these choices and their combinations, the choice can be difficult. 

What may help us get started is looking at the same task being done in different ways. Here, we build a host of models for the same classification task -- Node classification of the CiteSeer Dataset -- and explore what happens when we combine different models and modalities of information for the class prediction task. First, we have the "shallow" ML approaches, such as Random Forest, Naive Bayes, XGBoost to name a few. Second, we have the "Deep" Learning (DL) models that enable us to use more complex relationships in the data. Third, we have the Graph-based ML (Graph ML) approaches that incorporate link information within the  into the mix. Fourth, we also have Bayesian Machine Learning models. Finally, we can combine these models to see how well our "army" or ensemble of models does together.


<!-- In short, the choice is not uniquand requires trial-and-error with different models to find the best one from the pool. In this work, we run a set of experiments for the same classification task -- Node classification of the CiteSeer Dataset -- and compare their performance. We also see what happens if we stack or combine different models together. -->


## Goals:
    1. Compare the performance of different ML models and their combinations for class prediction.
    2. Compare different implementations (e.g., Tensorflow and PyTorch)


## List of Experiments

[**Section1**: "Shallow" ML algorithms](https://github.com/Roussel006/Many_MLs_for_Node_Classification/blob/main/Section_1_Shallow_MLs.ipynb)

In this section, we will build eight "shallow" ML classifiers. We will learn how to implement each of them using Scikit-Learn and save the models for later comparisons with other models. The models we will develop are:

    1a. Naive Bayes
    1b. XGBoost
    1c. Decision Trees
    1d. Random Forest
    1e. Gradient Boosting
    1f. CatBoost
    1g. LightGBM
    1h. Support Vector Machines (SVM)

**Section 2**: Deep Learning Models 

In this section, we build five DL models, using **Tensorflow**. Later, we will see how to build the same models to PyTorch and learn how to translate models between Tensorflow and PyTorch.

    2a. A simple dense model: with only pooling layer between the input and the output layers.
    2b. A deeper model with more layers
    2c. An Recurrent Neural Network (RNN) with Long Short Term Memory (LSTM)
    2d. An RNN with Gated Recurrent Unit (GRU)
    2e. A Convolutional Neural Network (CNN)

[**Section 3**: Machine Learning with Graphs](https://github.com/Roussel006/Many_MLs_for_Node_Classification/blob/main/Section_3_Machine_Learning_with_Graphs.ipynb)

In this section, we will build several Graph-based MLs. The graph in the CiteSeer Dataset is an **undirected, unweighted** graph, in essence, the simplest kind. As we will see, the incorporation of the edge information after setting each document as a node, will help us exceed the accuracies we had achieved with the shallow and the deep networks.

There are several types of ML with Graphs -- Feature Engineering (for Graph Characteristics as features), Graph Representation Learning, Graph Neural Networks, and more. Here, we will implement and compare some of the Graph Neural Networks (GNNs). Later, we may add some more models using Graph Representation Learning.

First, we will build two GNNs following the [Graph Neural Network Course](https://github.com/mlabonne/Graph-Neural-Network-Course) by Maxime Labonne. These two GNNs use both the text and the graph data available in the CiteSeer Dataset. 

Thereafter, we will build several variants of these two models, manipulating the input information to the GNNs. First, we take away the texts and only provide the information on the citations. Then, we keep only the text information and take away the graph information. This way, we will gain an understanding of the contributions of each stream of information to the accuracy of the models.

    3a. A Graph Convolutional Network (GCN)
    3b. A Graph Attention Network (GAT)

    ... (More experiments to be added)


<!-- **Graph Representation Learning**
Node Embeddings: Techniques like DeepWalk, node2vec, and LINE that learn low-dimensional representations of nodes.
Graph Embeddings: Methods like GraphSAGE and Graph Convolutional Networks (GCNs) that learn representations for entire graphs.
Edge Embeddings: Techniques that focus on learning representations for edges in a graph.

**Feature Engineering**
Manual Feature Extraction: Extracting features like node degree, clustering coefficient, and centrality measures.
Graph Kernels: Methods like Weisfeiler-Lehman Kernel and Shortest Path Kernel that measure similarity between graphs.
Graph Augmentation: Techniques that enhance graph data by adding or modifying nodes and edges.

**Graph Neural Networks (GNNs)**
Graph Convolutional Networks (GCNs): Neural networks that apply convolution operations on graphs.
Graph Attention Networks (GATs): GNNs that use attention mechanisms to weigh the importance of neighboring nodes.
Graph Recurrent Networks (GRNs): GNNs that incorporate recurrent neural network architectures for sequential data.
Graph Autoencoders: Unsupervised learning models that encode graph data into a latent space and then decode it back.

**Graph-Based Algorithms**:
PageRank: An algorithm used to rank nodes in a graph based on their importance.
Label Propagation: A semi-supervised learning algorithm that propagates labels through the graph.
Community Detection: Algorithms like Louvain and Girvan-Newman that identify clusters or communities within a graph.

**Graph Generative Models**
GraphRNN: A recurrent neural network-based model for generating graphs.
GraphVAE: A variational autoencoder for generating graphs.
GraphGAN: A generative adversarial network for generating graphs.

**Some Graph-Based Applications**
Social Network Analysis: Analyzing social networks to understand relationships and influence.
Recommendation Systems: Using graph-based techniques to recommend items or connections.
Molecular Graphs: Analyzing molecular structures for drug discovery and chemistry.
Knowledge Graphs: Representing and reasoning over knowledge in a structured form.

These are just a few examples of the diverse techniques and applications in graph-based ML. One can find more details in the [Hugging Face blog](https://huggingface.co/blog/intro-graphml). -->

A note on resources: 
    
1. A great open-source resource for ML with Graphs is the the CS224W course by Jure Lescovec at Stanford. The lecture slides, notebooks, and problem sets for several years are provided in entirety (Thanks a lot, Dr. Lescovec!), such as the Fall 2023 resource can be found here: https://snap.stanford.edu/class/cs224w-2023/. Lecture videos for particular years can be found on YouTube -- Isn't that just awesome? Here is the lecture series for 2021: https://youtu.be/JAB_plj2rbA?si=ujCiFl6HBFSqlGVf
2. The [Graph Neural Network Course](https://github.com/mlabonne/Graph-Neural-Network-Course) by [@maximelabonne](https://twitter.com/maximelabonne). Some of the codes here will be developed based on Chapter 2 of the course and this companion article: https://mlabonne.github.io/blog/gat/
3. A quick introduction: [Hugging Face blog](https://huggingface.co/blog/intro-graphml).

**Section 4**: Bayesian Machine Learning

Bayesian Machine Learning models are built on the principles of Bayesian statistics. In these models, we assume prior distributions over the model parameters, which represent our initial beliefs/knowledge about the parameters before seeing any data. As we train the model and observe data, we update these priors to obtain posterior distributions according to Bayes' theorem. The posterior distributions represent our updated beliefs about the parameters after incorporating the data.

A key advantage of Bayesian ML is that it provides uncertainty estimates directly. That is, for any prediction, we can quantify the confidence in that prediction by looking at the spread of the posterior distribution. In contrast, traditional Deep Learning models typically do not provide uncertainty estimates directly, rather require additional techniques such as Dropout Regularization or Ensemble methods. This advantage of Bayesian ML is particularly useful in applications where understanding the uncertainty is crucial (e.g., in medical diagnosis, financial forecasting, autonomous driving, or modeling human behavior).

A big thanks for this section goes to Michael J. Schoelles for his awesome course on Cognitive Modeling at Rensselaer Polytechnic Institute. This course and a related course -- Programming for Cognitive Science and AI -- by Dr. Schoelles built the foundation of my Machine Learning knowledge and helped me immensely in developing this tutorial.

<!-- A key advantage of Bayesian ML is that it provides uncertainty estimates directly. That is, for any prediction, we can quantify the confidence in that prediction by looking at the spread of the posterior distribution. This is particularly useful in applications where understanding the uncertainty is crucial, such as in medical diagnosis, financial forecasting, autonomous driving, or modeling human behavior.

In contrast, traditional Deep Learning models typically do not provide uncertainty estimates directly. These models are often deterministic; meaning that for a given input, they produce a single output without any measure of uncertainty. To estimate uncertainty in Deep Learning models, additional techniques are required, such as:

1. Dropout: A regularization technique where random neurons are dropped during training. At inference time, dropout can be used to create an ensemble of models, and the variance in their predictions can be used as an uncertainty estimate.

2. Ensemble Methods: Training multiple models independently and combining their predictions. The variance in the predictions of the ensemble members can be used to estimate uncertainty.

While these methods can provide some measure of uncertainty, they are often less straightforward and may not capture the full range of uncertainty as effectively as Bayesian methods.

In summary, Bayesian Machine Learning models offer a more natural and direct way to quantify uncertainty in predictions, which can be a significant advantage over other Machine Learning and Deep Learning methods in many applications. -->

**Section 5**: Ensembling or Stacking different models

Here, we will try different combinations of all the models we have built, to see which ones are the best ones.


<!-- ## More options:

PyMC3 models -- Bayesian Learning models. Gaussian processes, 
Unsupervised methods (e.g., Spectral Clustering) -->
