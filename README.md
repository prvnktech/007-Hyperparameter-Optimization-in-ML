# Hyperparameter Optimization of Machine Learning Algorithms

To fit a machine learning model into different problems, its hyper-parameters must be tuned. Selecting the best hyper-parameter configuration for machine learning models has a direct impact on the model's performance. 


### Quick Navigation
**Section 3**: Important hyper-parameters of common machine learning algorithms  
**Section 4**: Hyper-parameter optimization techniques introduction  
**Section 5**: How to choose optimization techniques for different machine learning models  
**Section 6**: Common Python libraries/tools for hyper-parameter optimization  
**Section 7**: Experimental results (sample code in "HPO_Regression.ipynb" and "HPO_Classification.ipynb")  
**Section 8**: Open challenges and future research directions  
**Summary table for Sections 3-6**: Table 2:  A comprehensive overview of common ML models, their hyper-parameters, suitable optimization techniques, and available Python libraries  
**Summary table for Sections 8**: Table 10:  The open challenges and future directions of HPO research  

## Implementation
Sample code for hyper-parameter optimization implementation for machine learning algorithms is provided in this repository.  

### Sample code for Regression problems  
[HPO_Regression.ipynb](https://github.com/LiYangHart/Hyperparameter-Optimization-of-Machine-Learning-Algorithms/blob/master/HPO_Regression.ipynb)   
**Dataset used:** [Boston-Housing](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html)   
### Sample code for Classification problems  
[HPO_Classification.ipynb](https://github.com/LiYangHart/Hyperparameter-Optimization-of-Machine-Learning-Algorithms/blob/master/HPO_Classification.ipynb)   
**Dataset used:** [MNIST](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html#sklearn.datasets.load_digits)   

### Machine Learning & Deep Learning Algorithms  
* Random forest (RF)
* Support vector machine (SVM)
* K-nearest neighbor (KNN)  
* Artificial Neural Networks (ANN)

### Hyperparameter Configuration Space  
|     ML Model          |     Hyper-parameter      |     Type           |     Search Space                            |
|-----------------------|--------------------------|--------------------|---------------------------------------------|
|     RF Classifier     |     n_estimators         |     Discrete       |     [10,100]                                |
|                       |     max_depth            |     Discrete       |     [5,50]                                  |
|                       |     min_samples_split    |     Discrete       |     [2,11]                                  |
|                       |     min_samples_leaf     |     Discrete       |     [1,11]                                  |
|                       |     criterion            |     Categorical    |     'gini', 'entropy'                     |
|                       |     max_features         |     Discrete       |     [1,64]                                  |
|     SVM Classifier    |     C                    |     Continuous     |     [0.1,50]                                |
|                       |     kernel               |     Categorical    |     'linear', 'poly', 'rbf', 'sigmoid'    |
|     KNN Classifier    |     n_neighbors          |     Discrete       |     [1,20]                                  |
|     ANN Classifier    |     optimizer         |     Categorical       |     'adam', 'rmsprop', 'sgd'                                |
|                       |     activation            |     Categorical       |     'relu', 'tanh'                                  |
|                       |     batch_size    |     Discrete       |     [16,64]                                  |
|                       |     neurons     |     Discrete       |     [10,100]                                  |
|                       |     epochs            |     Discrete    |     [20,50]                     |
|                       |     patience         |     Discrete       |     [3,20]                                  |
|     RF Regressor      |     n_estimators         |     Discrete       |     [10,100]                                |
|                       |     max_depth            |     Discrete       |     [5,50]                                  |
|                       |     min_samples_split    |     Discrete       |     [2,11]                                  |
|                       |     min_samples_leaf     |     Discrete       |     [1,11]                                  |
|                       |     criterion            |     Categorical    |     'mse', 'mae'                          |
|                       |     max_features         |     Discrete       |     [1,13]                                  |
|     SVM Regressor     |     C                    |     Continuous     |     [0.1,50]                                |
|                       |     kernel               |     Categorical    |     'linear', 'poly', 'rbf', 'sigmoid'    |
|                       |     epsilon              |     Continuous     |     [0.001,1]                               |
|     KNN Regressor     |     n_neighbors          |     Discrete       |     [1,20]                                  |
|     ANN Regressor     |     optimizer         |     Categorical       |     'adam', 'rmsprop'                              |
|                       |     activation            |     Categorical       |     'relu', 'tanh'                                  |
|                       |     loss            |     Categorical       |     'mse', 'mae'                                  |
|                       |     batch_size    |     Discrete       |     [16,64]                                  |
|                       |     neurons     |     Discrete       |     [10,100]                                  |
|                       |     epochs            |     Discrete    |     [20,50]                     |
|                       |     patience         |     Discrete       |     [3,20]                                  |

### HPO Algorithms  
* Grid search
* Random search
* Hyperband
* Bayesian Optimization with Gaussian Processes (BO-GP)
* Bayesian Optimization with Tree-structured Parzen Estimator (BO-TPE)
* Particle swarm optimization (PSO)
* Genetic algorithm (GA)  

### Requirements  
* Python 3.5+ 
* [Keras](https://keras.io/) 
* [scikit-learn](https://scikit-learn.org/stable/)  
* [hyperband](https://github.com/thuijskens/scikit-hyperband)  
* [scikit-optimize](https://github.com/scikit-optimize/scikit-optimize)  
* [hyperopt](https://github.com/hyperopt/hyperopt)  
* [optunity](https://github.com/claesenm/optunity)  
* [DEAP](https://github.com/rsteca/sklearn-deap)  
* [TPOT](https://github.com/EpistasisLab/tpot)  

## Contact-Info
Please feel free to contact me for any questions.
* Email: [praveen.kumar@zummitlabs.com](mailto:praveen.kumar@zummitlabs.com)
* LinkedIn: [Praveen Kumar](https://www.linkedin.com/in/praveen-zummit/)  
