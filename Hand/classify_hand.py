
import joblib
import numpy as np

default_model_path = "hand_model.joblib"
__model = None

def classify_hand(coordinates, model = None):
    # Ensure the input coordinates is a 2D array (1 sample, 63 features)
    if model is None:
        global __model
        if __model is None:
            __model = joblib.load(default_model_path)
        model = __model

    coordinates = np.array(coordinates).reshape(1, -1)
    # Make prediction
    prediction = model.predict(coordinates)
    return prediction[0]

def predict_hand_prob(coordinates, model=None):
    # Ensure the input coordinates is a 2D array (1 sample, 63 features)
    if model is None:
        global __model
        if __model is None:
            __model = joblib.load(default_model_path)
        model = __model

    coordinates = np.array(coordinates).reshape(1, -1)
    # Make prediction probabilities
    probabilities = model.predict_proba(coordinates)
    return probabilities[0].tolist()

if __name__ == '__main__':
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    import pandas as pd
    
    # Load the training and test data
    train_data = pd.read_csv('hand_data_train.csv')
    test_data = pd.read_csv('hand_data_test.csv')
    
    # Preprocess the data
    def preprocess_data(df):
        labels = df.iloc[:, 0]
        features = df.iloc[:, 1:]
        x_coords = features.iloc[:, :21]
        y_coords = features.iloc[:, 21:42]
        z_coords = features.iloc[:, 42:]
        x_coords_centered = x_coords.sub(x_coords.mean(axis=1), axis=0)
        y_coords_centered = y_coords.sub(y_coords.mean(axis=1), axis=0)
        z_coords_centered = z_coords.sub(z_coords.mean(axis=1), axis=0)
        centered_features = pd.concat([x_coords_centered, y_coords_centered, z_coords_centered], axis=1)
        return labels, centered_features

    train_labels, train_features = preprocess_data(train_data)
    test_labels, test_features = preprocess_data(test_data)

    # Train the model
    clf = RandomForestClassifier(random_state=420)
    clf.fit(train_features, train_labels)
    
    # Save the trained model
    joblib.dump(clf, 'hand_model.joblib')
    
    # Reload the model
    # model = joblib.load('hand_model.joblib')
    
    # # Define the classify_hand function
    # def classify_hand(coordinates, model):
    #     coordinates = np.array(coordinates).reshape(1, -1)
    #     prediction = model.predict(coordinates)
    #     return prediction[0]
    model = None

    # Test classify_hand function on the test set
    test_predictions = [classify_hand(row, model) for row in test_features.values]
    test_accuracy = accuracy_score(test_labels, test_predictions)
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
