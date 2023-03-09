# Random-Forest-in-Python
There has never been a better time to get into machine learning. With the learning resources available online, free open-source tools with implementations of any algorithm imaginable, and the cheap availability of computing power through cloud services such as AWS, machine learning is truly a field that has been democratized by the internet. Anyone with access to a laptop and a willingness to learn can try out state-of-the-art algorithms in minutes. With a little more time, you can develop practical models to help in your daily life or at work (or even switch into the machine learning field and reap the economic benefits). This post will walk you through an end-to-end implementation of the powerful random forest machine learning model. It is meant to serve as a complement to my conceptual explanation of the random forest, but can be read entirely on its own as long as you have the basic idea of a decision tree and a random forest. A follow-up post details how we can improve upon the model built here.
Before we jump right into programming, we should lay out a brief guide to keep us on track. The following steps form the basis for any machine learning workflow once we have a problem and model in mind:

1.State the question and determine required data
2.Acquire the data in an accessible format
3.Identify and correct missing data points/anomalies as required
4.Prepare the data for the machine learning model
5.Establish a baseline model that you aim to exceed
6.Train the model on the training data
7.Make predictions on the test data
8.Compare predictions to the known test set targets and calculate performance metrics
9.If performance is not satisfactory, adjust the model, acquire more data, or try a different modeling technique
10.Interpret model and report results visually and numerically
