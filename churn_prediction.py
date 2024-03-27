import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras import layers
from keras import utils

# Load data
df = pd.read_csv('waze_dataset.csv')
print(df.info())

# Null values
null_df = df[df['label'].isnull()]
print(null_df.describe())
# Get count of null values by device
print(null_df['device'].value_counts())

# % of iPhone nulls and Android nulls
print(null_df['device'].value_counts(normalize=True))

# % of iPhone users and Android users in full dataset
print(df['device'].value_counts(normalize=True))


# counts of churned vs. retained
print(df['label'].value_counts())

print(df['label'].value_counts(normalize=True))

# median values for churned and retained users
df.groupby('label').median(numeric_only=True)

# Preprocess the data
# Encoding categorical variables (if any)
le_device = LabelEncoder()
df['device'] = le_device.fit_transform(df['device'])

# Encoding the label
le_label = LabelEncoder()
df['label'] = le_label.fit_transform(df['label'])

# Normalize/Standardize numerical features
scaler = StandardScaler()
numerical_features = df.select_dtypes(include=np.number).columns.tolist()
numerical_features.remove('label')  # Assuming 'label' is the target variable
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Split the dataset into features and target variable
X = df[numerical_features].values
y = df['label'].values

def to_categorical(y, num_classes=None):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=np.float32)
    categorical[np.arange(n), y] = 1
    return categorical

# Convert labels to categorical
y = to_categorical(y)

# Reshape X to be [samples, time steps, features] which is required for RNN
X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create the RNN model
model = keras.Sequential()
model.add(layers.SimpleRNN(50, input_shape=(1, X.shape[2]), activation='relu'))
model.add(layers.Dense(y.shape[1], activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print('Test accuracy: ', accuracy)



