import pandas as pd

# Load the provided train and test datasets
train_data = pd.read_csv('hand_data_train.csv')
test_data = pd.read_csv('hand_data_test.csv')


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Preprocessing function to normalize the x, y, z coordinates
def preprocess_data(df):
    labels = df.iloc[:, 0]
    features = df.iloc[:, 1:]
    
    # Splitting the features into x, y, z coordinates
    x_coords = features.iloc[:, :21]
    y_coords = features.iloc[:, 21:42]
    z_coords = features.iloc[:, 42:]
    
    # Subtracting the mean of each set of coordinates from their respective values
    x_coords_centered = x_coords.sub(x_coords.mean(axis=1), axis=0)
    y_coords_centered = y_coords.sub(y_coords.mean(axis=1), axis=0)
    z_coords_centered = z_coords.sub(z_coords.mean(axis=1), axis=0)
    
    # Concatenating the centered x, y, z coordinates back together
    centered_features = pd.concat([x_coords_centered, y_coords_centered, z_coords_centered], axis=1)
    
    return labels, centered_features

# Preprocess both the training and test datasets
train_labels, train_features = preprocess_data(train_data)
test_labels, test_features = preprocess_data(test_data)

# Initialize and train the Random Forest classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(train_features, train_labels)

# Make predictions and calculate accuracy on both training and test datasets
train_predictions = clf.predict(train_features)
test_predictions = clf.predict(test_features)

train_accuracy = accuracy_score(train_labels, train_predictions)
test_accuracy = accuracy_score(test_labels, test_predictions)

print(train_accuracy, test_accuracy)
