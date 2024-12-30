# Titanic-Survival-Prediction-using-Bagging
## Overview
This project implements a bagging ensemble method using decision tree stumps to predict the survival of passengers on the Titanic. The ensemble model consists of 100 decision tree stumps trained on bootstrap samples of the training data. This implementation does not use pre-built Python libraries for bagging.

## Prerequisites
Before running the project, make sure you have the following installed:
- Python 3.6 or newer
- pandas
- scikit-learn

You can install the required Python packages using pip:
pip install pandas scikit-learn

Dataset
The dataset consists of two CSV files:
titanikData.csv: Training data including features and survival labels.
titanikTest.csv: Test data used to evaluate the model.
Each passenger record includes the following features:

pclass: Passenger class (1st, 2nd, 3rd)
age: Age of the passenger
gender: Gender of the passenger (male, female)
survived: Survival status (yes, no) — present only in training data.
Files
bagging_titanic.py: Python script that includes all the functions and model training logic.
README.md: This file, providing an overview and instructions for running the project.

Running the Project
To run the project, follow these steps:
Clone the repository to your local machine.
Navigate to the directory containing the project files.

Execute the script using Python:
python bagging_titanic.py

Methodology
The project follows these steps:

Data Loading: Load the training and testing data from CSV files.
Bootstrap Sampling: Generate 100 bootstrap samples from the training data.
Decision Tree Stumps: Train a decision tree stump on each bootstrap sample.
Majority Voting: Use the trained stumps to predict survival on the test dataset by majority voting.
Evaluation: Calculate the accuracy of the ensemble model by comparing the predicted results with the actual survival statuses in the test data.
Output
The script will print the accuracy of the ensemble model to the console and display the predictions for each passenger in the test data.

License
This project is released under the MIT License.
