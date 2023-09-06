# AppliedMachineLearning
Welcome to our Applied Machine Learning Repository! These projects were completed under the guidance of the 🤖 Applied Machine Learning 🤖 Course at McGill (COMP 551). While the projects had general guidelines, we were encouraged to go above and beyond 🚀, and to explore the implemenations of every model in various ways, a large portion of every grade we recieved. Each of the 4 projects is summarized below, and can be found in the corresponding folder, along with its IEEE report describing our research. 

## Brief Summary 📘📖
- Implemented linear regression models in NumPy with several gradient descent methods and L1 Regularization, to predict two output variables from an 8-feature input quantitative energy dataset, achieving a MSE of less than 10 for all models 🔥
- Performed logistic regression on a bankruptcy dataset to achieve an accuracy of 94% - 100% 🍉
- Designed an MLP model in NumPy to classify images from the CFAR-10 dataset, with different activation functions, variable depth, hidden units, ADAM gradient descent, and L2 regularization. 54% accuracy, higher than several papers ⬆️
- Constructed a 2 layer CNN with max pooling with TensorFlow to achieve 72.5% accuracy on the CFAR-10 dataset. Performed a transfer learning exercise with ResNet50V2 architecture to achieve 86.2% accuracy 🏹
- Created a Bayesian Naïve Bayes Model in NumPy to achieve 85.3% accuracy in sentiment classification task of the IMDB review dataset, beating several published papers, narrowly behind the accuracy of our finetuned BERT model (88%) 🥇

Each of the projects, their corresponding papers, and the results of each paper are summarized in more detail below

## Linear and Logisitic Regression 🏠 🌩️ (92.6%)
This project was centred around the introduction to common Machine Learning models. Its principal task was the implementation of two models: Linear Regression, and Logistic Regression with Gradient Descent. Our team implemented these models in Python using several core machine learning techniques: 

- Analytical linear regression
- Fully batched gradient descent
- Mini-batch stochastic gradient descent
- Lasso regression for L1 regularization
- Adaptive Moment Estimation (ADAM)

We performed linear regression on an eight-feature quantitative energy dataset, using our model to predict two output variables, and logistic regression on a six-feature categorical bankruptcy dataset, using this model to classify test instances with the correct label. We also ran several experiments on each implementation, allowing us to demonstrate that all our models were highly accurate, with mean squared error values less than 10 for all linear regression models, and an accuracy of 94 - 100\% for the logistic regression models. Further analysis helped us determine the most salient features of each collection, as well as the most appropriate hyper-parameters for the various algorithms. In this project and its corresponding paper, we studied the performance of each approach on two fixed datasets and analyze their strengths and potential shortcomings.

## Image Classification 📷🧠 (100%)

Neural networks, mimicking the neural pathways in the brain on a far smaller scale, have an extensive range of uses, including image classification. In this project, multiple neural networks are trained to perform image classification and tested against the CIFAR-10 dataset, which consists of 60,000 color images categorized into 10 different classes. The first neural network explored is a Multi-Layer Perceptron, implemented from the ground up to allow for any number of hidden layers and hidden units, as well as ReLU, Tanh or LeakyReLU activation functions, ending with a softmax output layer. Our implementation primarily uses L2 regularization to prevent overfitting with both Mini-batch and Adam gradient descent. Our exploration of model depth, activation functions, regularization, data normalization, hyper-parameter values and gradient descent optimizers discovers the best model to be 2 layer MLP with 256 hidden units at each layer, ReLU activation functions, and L2 regularization. This model has a test accuracy of 54\% after being trained on 20 epochs of data with a batch size of 32 and a learning rate of 0.013 or 0.0004 for Mini-batch and Adam gradient descent respectively. 

The second model investigated is a Convolutional Neural Network (CNN), implemented using TensorFlow libraries. When hyper-parameter tuning is performed and max pooling added, the CNN has 72\% test accuracy. Finally, a transfer learning exercise is performed on the dataset using the ResNet50V2 architecture. Using the pre-trained weights on the ImageNet dataset and training fully connected layers on top of the frozen model, a test accuracy of a 86\% is obtained. These major results are summarized in the table below
     
  | Model                    | Test accuracy (%) |
  |--------------------------|-------------------|
  | 2 Hidden Layers MLP (ReLU) | 54.1              |
  | Benchmark CNN            | 72.5              |
  | 1 Layer Custom ResNet-50V2 | 86.2              |

## Sentiment Classification 🎥💙(100%)

This project and its paper contrasts the power of simplicity and complexity in machine learning models by comparing a set of simple models - Normal, Bayesian and Multinomial Naive Bayes models - with the complex pre-trained BERT transformer network. All of these models have been either designed or adapted to perform sentiment classification on a benchmark dataset of positive and negative movie reviews from IMDB. For the Naive Bayes models, feature selection was performed on the training dataset, and models were compared with optimal values for smoothing parameters $\alpha$ and $\beta$. With these parameters, the best Naive Bayes model was the Bayesian Naive Bayes model with $\alpha$ and $\beta$ set to 0.2, giving a training accuracy of  86.3\% and a test accuracy of 85.3\%. For the BERT model, a transfer learning exercise was performed on the same IMDB dataset with a BERT variation, ALBERT, A Lite BERT. The training procedure was inspired from an official TensorFlow tutorial that follows the practices outlined in the original BERT paper. After hyperparameter tuning, the fine-tuned ALBERT model attained a test accuracy of 88.8\%. Final Results are summarized in the table below:

| Model                 | Test Accuracy (%) |
|-----------------------|-------------------|
| Bayesian Naive Bayes  | 85.3              |
| BERT                  | 88.8              |

## Reproducibility Challenge ⚽🏆 (93.1%)

This project and its paper explores a random forest approach to predicting the outcome of the FIFA World Cup in 2018, and the impact of both the complexity of data available and the effect of repeated simulations on model performance. The approach is based on a paper which uses a combination of random forests, poisson regression and ranking methods to predict the expected number of goals a team will score in a given game, which is then used to determine the winner. However, this approach is beyond our time and complexity constraints. To explore three of the papers main claims, our paper implements a simpler model that uses random forests to predict the most probable winning team. We explore three claims the paper makes both implicitly and explicitly:

- More complex processed features are better data separators and therefore more valuable than less complex and raw features for the a random forest model
- Repeated simulation of the world cup stages leads to a model with higher predictive power than a single simulation
- A combination of random forest, Poisson regression and ranking methods have a higher predictive power than random forests alone.

The last claim was explored by comparing the results an implementation of a simpler model with the results of the more complex one in the paper, since we could time and computational constrains meant that we could not implement the complex model. The two final results are shown below: 

### A Simpler Model 
![mean_bracket](https://github.com/sjavaheri/AppliedMachineLearning/assets/97904673/1e601d2e-5054-4d37-a0a8-354e305b44b4)

### More Complex Model
![report_bracket](https://github.com/sjavaheri/AppliedMachineLearning/assets/97904673/42ff00f5-926a-464f-916e-1a259e0f33ad)



