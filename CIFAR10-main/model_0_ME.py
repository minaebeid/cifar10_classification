import tensorflow as tf
from sklearn.metrics import confusion_matrix , classification_report
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
X_train.shape
X_test.shape

y_train = y_train.reshape(-1,)

classes = ["airplane", "automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
def plot_sample(X, y, index):
    plt.imshow(X[index])
    plt.xlabel(classes[y[index]])

X_train = X_train /255
X_test = X_test / 255

Model = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

Model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

Model.fit(X_train, y_train, epochs=10)

y_prediction = Model.predict(X_test)
y_prediction_classes = [np.argmax(element) for element in y_prediction]
print("Report: \n", classification_report(y_test, y_prediction_classes))