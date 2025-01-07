import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Load the training and testing data
train_data = pd.read_csv('titanikData.csv')
test_data = pd.read_csv('titanikTest.csv', names=['pclass', 'age', 'gender', 'survived'], header=0)

# One-hot encoding of categorical variables for training and testing data
encoder = OneHotEncoder()
encoded_train_features = encoder.fit_transform(train_data[['pclass', 'age', 'gender']]).toarray()
train_data_encoded = pd.DataFrame(encoded_train_features, columns=encoder.get_feature_names_out(['pclass', 'age', 'gender']))
train_data_encoded['survived'] = train_data['survived'].map({'yes': 1, 'no': 0})

encoded_test_features = encoder.transform(test_data[['pclass', 'age', 'gender']]).toarray()
test_data_encoded = pd.DataFrame(encoded_test_features, columns=encoder.get_feature_names_out(['pclass', 'age', 'gender']))
test_data_encoded['survived'] = test_data['survived'].map({'yes': 1, 'no': 0})

def bootstrap_samples(data, n_samples):
    return [data.sample(n=len(data), replace=True) for _ in range(n_samples)]

def build_stumps(data, label):
    trees = []
    for sample in bootstrap_samples(data, 100):
        stump = DecisionTreeClassifier(max_depth=1)  # Stump has max depth of 1
        features = sample.drop(columns=[label])
        labels = sample[label]
        stump.fit(features, labels)
        trees.append(stump)
    return trees

def majority_voting(trees, data):
    predictions = []
    for tree in trees:
        predictions.append(tree.predict(data.drop(columns=['survived'])))
    predictions = np.array(predictions)
    final_predictions = np.apply_along_axis(lambda x: np.bincount(x, minlength=2).argmax(), axis=0, arr=predictions)
    return final_predictions

trees = build_stumps(train_data_encoded, 'survived')
test_predictions = majority_voting(trees, test_data_encoded)
test_data_encoded['predicted'] = test_predictions
accuracy = np.mean(test_data_encoded['survived'] == test_data_encoded['predicted'])

def plot_data(data):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.countplot(x='gender', hue='survived', data=data)
    plt.title('Survival Count by Gender')
    plt.subplot(1, 2, 2)
    sns.countplot(x='pclass', hue='survived', data=data)
    plt.title('Survival Count by Passenger Class')
    plt.show()

# Displaying the final results in a visual table
print(test_data_encoded[['survived', 'predicted']])
print(f'Accuracy: {accuracy * 100:.2f}%')
plot_data(train_data)
