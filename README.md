# MasterThesis - Kilian Bock
## Abstract
One of the main challenges in applying learning techniques to classification problems is the large amount of labeled training data required.
Especially for microstructural images in material science training data is very expensive in terms of time, effort, and cost to come by.
In this study, several active learning approaches to tackle the problem are investigated.
Rather than accepting random training examples, active learning iteratively selects unlabeled examples for the user to label, so that the user's energy is focused on labeling the most "informative" examples. The examined strategies belong to Uncertainty sampling, in which the algorithm selects unlabeled examples that it finds hardest to classify, Hypothesis space search, in which selections are made according to the uncertainty in predictions of several machine learning models and a method based on distance measures in the feature space. Further, the impact of the choice of learning algorithm was investigated. Therefore a random forest model and a neural network were compared on various generated feature sets.\\
Specifically, results for those strategies were demonstrated for two different classification scenarios for a fractographic data set. Namely, those scenarios were fracture classification and material affiliation, which were a binary and multi-class cases, respectively. The proposed methods give large reductions in the number of training examples required over random selection to achieve similar classification accuracy.


# Thesis
You can find the whole thesis [here](https://github.com/Kiliantae/Master-Thesis/blob/main/Masterthesis.pdf).
All code that we used for the thesis is available in this repository.
