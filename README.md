# MachineLearning
This is a collection of 5 different supervised learning algorithms including decision trees, neural networks, k-NN, decision trees with boosting, and support vector machines. 

The algorithms are run on two different datasets found from Kaggle

## Datasets
The easiest way to access the datasets is to install the kaggle package. If all else fails, feel free to download them manually from the links provided below, and place them in the datasets folder. 

```
pip install kaggle
kaggle configure
```

You can set the configurations with the API key from the kaggle account. 

To download the datasets in the database folder (where they will be used to run the code):

```
kaggle datasets download -d datasnaek/chess -p /datasets
kaggle datasets download -d sakshigoyal7/credit-card-customers -p /datasets
```

Dataset 1:
https://www.kaggle.com/datasets/datasnaek/chess

Dataset 2:
https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers


## Running Code
### Environment setup
To run the following code, you will need to have python installed. The specific version used for development was python 3.11.5. 

### Installing Packages
All the required packages are placed in the folder called `requirements.txt`. 

To install all packages run the following:
```
pip install -r requirements.txt
```

### Entry points: Notebooks(.ipynb)
There are two notebooks for each dataset. Both notebooks contain all the steps from preprocessing to running every algorithm on the datasets. You will notice that each dataset undergoes different data preprocessing methods to achieve more meaningful insights to the models which are made from them. In addition, the notebooks call the functions which prepare statistical data and graphs that are used for analysis.

## Files
1. `ChurnersNotebook.ipynb`: Includes all the code to run to clean the data from the BankChurners dataset, and run all five algorithms on them. This includes producing graphs and plots crucials for anlyzing the models.
2. `ChessNotebook.ipynb`: Includes all the code to run to clean the data from the ChessGames dataset, and run all five algorithms on them. This includes producing graphs and plots crucials for anlyzing the models.
3. `constants.py`: Includes all the constants used globally to ensure reproducibility of the models. I.e. splitting the test and training data using random states.
4. `dt.py`: Includes all the algorithms required for running both decision trees and decision trees with boosting.
5. `knn.py`: Includes all the algorithms required for running the k-Nearest Neighbors models.
6. `nn.py`: Includes all the algorithms used to develop the models for neural networks.
7. `svm.py`: Includes the algorithms used to run the support vector machine algorithms.
8. `graphs.py`: Includes the functions to prepare and display graphs crucial for the analysis of the models.
9. `stats.py`: Includes the functions to provide statistical data on the models used for analysis. 