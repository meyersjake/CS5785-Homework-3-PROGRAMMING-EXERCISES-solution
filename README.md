# CS5785-Homework-3-PROGRAMMING-EXERCISES-solution

Download Here: [CS5785 Homework 3 PROGRAMMING EXERCISES solution](https://jarviscodinghub.com/assignment/homework-3-programming-exercises-solution/)

For Custom/Original Work email jarviscodinghub@gmail.com/whatsapp +1(541)423-7793

1. Sentiment analysis of online reviews.
In this assignment you will use several machine learning techniques from the class to identify
and extract subjective information in online reviews. Specifically, the task for this assignment is
sentiment analysis. According to Wikipedia, sentiment analysis aims to determine the attitude of
a speaker or a writer with respect to some topic or the overall contextual polarity of a document.
It has been shown that people’s attitudes are largely manifested in the language they adopt. This
assignment will walk you through the mystery and help you better understand our posts online!
Important: Use your own implementations for this assignment. Any K-means, bag-of-words or
PCA implementation is NOT allowed.
(a) Download Sentiment Labelled Sentences Data Set. There are three data files under the root
folder. yelp_labelled.txt, amazon_cells_labelled.txt and imdb_labelled.txt. Parse each file
with the specifications in readme.txt. Are the labels balanced? If not, what’s the ratio between
the two labels? Explain how you process these files.
(b) Pick your preprocessing strategy. Since these sentences are online reviews, they may contain significant amounts of noise and garbage. You may or may not want to do one or all of
the following. Explain the reasons for each of your decision (why or why not).
• Lowercase all of the words.
• Lemmatization of all the words.
• Strip punctuation.
• Strip the stop words, e.g., “the”, “and”, “or”.
• Something else? Tell us about it.
(c) Split training and testing set. In this assignment, for each file, please use the first 400 instances for each label as the training set and the remaining 100 instances as testing set. In
total, there are 2400 reviews for training and 600 reviews for testing.
(d) Bag of Words model. Extract features and then represent each review using bag of words
model, i.e., every word in the review becomes its own element in a feature vector. In order to
do this, first, make one pass through all the reviews in the training set (Explain why we can’t
use testing set at this point) and build a dictionary of unique words. Then, make another pass
through the review in both the training set and testing set and count up the occurrences of
each word in your dictionary. The ith element of a review’s feature vector is the number of
occurrences of the ith dictionary word in the review. Implement the bag of words model and
report feature vectors of any two reviews in the training set.
(e) Pick your postprocessing strategy. Since the vast majority of English words will not appear in
most of the reviews, most of the feature vector elements will be 0. This suggests that we need
a postprocessing or normalization strategy that combats the huge variance of the elements
in the feature vector. You may want to use one of the following strategies. Whatever choices
you make, explain why you made the decision.
• log-normalization. For each element of the feature vector x, transform it into f (x) =
l og (x +1).
• l1 normalization. Normalize the l1 norm of the feature vector, xˆ =
x
| x |
.
2
• l2 normalization. Normalize the l2 norm of the feature vector, xˆ =
x
kxk
.
• Standardize the data by subtracting the mean and dividing by the variance.
(f ) Training set clustering. For the feature vectors you computed from (e). Implement K-means
algorithm to divide the training set into 2 clusters (i.e., K = 2). Report the centers of both clusters. Inspect the clustering results and the labels associated the each instance, and evaluate
the performance of your K-means algorithm and bag of words feature representation.
(g) Sentiment prediction. Train a logistic regression model (you can use existing packages here)
on the training set and test on the testing set. Report the classification accuracy and confusion matrix. Inspecting the weight vector of the logistic regression, what are the words that
play the most important roles in deciding the sentiment of the reviews?
(h) N-gram model. Similar to the bag of words model, but now you build up a dictionary of ngrams, which are contiguous sequences of words. For example, “Alice fell down the rabbit
hole” would then map to the 2-grams sequence: [“Alice fell”, “fell down”, “down the”, “the
rabbit”, “rabbit hole”], and all five of those symbols would be members of the n-gram dictionary. Try n = 2, repeat (d)-(g) and report your results.
(i) PCA for bag of words model. The features in the bag of words model have large redundancy.
Implement PCA to reduce the dimension of features calculated in (e) to 10, 50 and 100 respectively. Using these lower-dimensional feature vectors and repeat (f ), (g). Report corresponding clustering and classification results. (Note: You should implement PCA yourself,
but you can use numpy.svd or some other SVD package. Feel free to double-check your PCA
implementation against an existing one)
(j) Algorithms comparison and analysis. According to the above results, compare the performances of bag of words, 2-gram and PCA for bag of words. Which method performs best in
the prediction task and why? What do you learn about the language that people use in online reviews (e.g., expressions that will make the posts positive/negative)? Hint: Inspect the
clustering results and the weights learned from logistic regression.
2. EM algorithm and implementation
(a) The parameters of Gaussian Mixture Model (GMM) can be estimated via the EM algorithm.
Show that the alternating algorithm for K-means (in Lec. 11) is a special case of the EM algorithm and show the corresponding objective functions for E-step and M-step.
(b) Download the Old Faithful Geyser Dataset. The data file contains 272 observations of (eruption time, waiting time). Treat each entry as a 2 dimensional feature vector. Parse and plot all
data points on 2-D plane.
(c) Implement a bimodal GMM model to fit all data points using EM algorithm. Explain the reasoning behind your termination criteria. For this problem, we assume the covariance matrix
is spherical (i.e., it has the form of σ
2
I) and you can randomly initialize Gaussian parameters.
For evaluation purposes, please submit the following figures:
• Plot the trajectories of two mean vectors in 2 dimensions (i.e., coordinates vs. iteration).
• Run your program for 50 times with different initial parameter guesses. Show the distribution of the total number of iterations needed for algorithm to converge.
3
(d) Repeat the task in (c) but with the initial guesses of the parameters generated from the following process:
• Run a K-means algorithm over all the data points with K = 2 and label each point with
one of the two clusters.
• Estimate the first guess of the mean and covariance matrices using maximum likelihood
over the labeled data points.
Compare the algorithm performances of (c) and (d).
WRITTEN EXERCISES
You can find links to the textbooks for our class on the course website.
Submit the answers to these questions along with your writeup as a single .pdf file. We do recommend you to type your solutions using LaTeX or other text editors, since hand-written solutions are
often hard to read. If you handwrite them, please be legible!
1. HTF Exercise 14.2 (Gaussian mixture model and EM algorithm)
2. HTF Exercise 14.8 (Procrustes problem.)
3. HTF Exercise 14.11 (Multidimensional Scaling.)

