import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 1: Load the pickle file containing the datasets
with open('./data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

# Step 2: Check the structure of the data
print("Data structure:", type(data_dict['data']))  # Should be a list
print("Number of datasets:", len(data_dict['data']))  # Should be 287 datasets
print("First dataset length:", len(data_dict['data'][0]))  # Length of the first dataset (42)

# Step 3: Padding or truncating the feature vectors to make them uniform
max_length = max(len(dataset) for dataset in data_dict['data'])  # Find the longest dataset

# Padding shorter datasets with zeros
padded_data = []
for dataset in data_dict['data']:
    if len(dataset) < max_length:
        # Pad with zeros if the length is shorter than the longest dataset
        padded_data.append(dataset + [0] * (max_length - len(dataset)))
    else:
        padded_data.append(dataset)

# Step 4: Convert to NumPy array
data = np.array(padded_data)  # Now all datasets should have the same length
labels = np.array(data_dict['labels'])  # Assuming labels are compatible

# Step 5: Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Step 6: Train the RandomForest model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Step 7: Evaluate the model's performance
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print(f'{score * 100:.2f}% of samples were classified correctly!')

# Step 8: Save the trained model to a file
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
