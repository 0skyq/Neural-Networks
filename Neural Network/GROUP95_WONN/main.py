#Roll no : 23AT61R04
#Name : Thummalabavi Sankshay reddy
#project name : WONN

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def train(X, Y, in_units, hid_units, out_units, learning_rate, batch_size):
    w_hid = np.random.uniform(low=0, high=0.01, size=(in_units, hid_units))
    w_out = np.random.uniform(low=0, high=0.01, size=(hid_units, out_units))

    num_batches = len(X) // batch_size
    losses = []

    for epoch in range(500):
        epoch_loss = 0
        for batch in range(num_batches):
            batch_start = batch * batch_size
            batch_end = min((batch + 1) * batch_size, len(X))

            batch_X = X.iloc[batch_start:batch_end]
            batch_Y = Y.iloc[batch_start:batch_end]

            grad_w_hid = np.zeros_like(w_hid)
            grad_w_out = np.zeros_like(w_out)

            for index, arr in batch_X.iterrows():
                arr = np.array(arr).reshape(1, -1)
                act_output = np.zeros(out_units)
                act_output[batch_Y.loc[index] - 1] = 1

                hid_output = sigmoid(np.dot(arr, w_hid))
                output = relu(np.dot(hid_output, w_out))

                e_out = (output - act_output) * output * (1-output)
                grad_w_out += np.dot(hid_output.T, e_out)

                e_hid = np.dot(e_out, w_out.T) * hid_output * (1 - hid_output)
                grad_w_hid += np.dot(arr.T, e_hid)

                loss = mean_squared_error(act_output, output.squeeze())
                epoch_loss += loss

            w_out -= learning_rate * grad_w_out / batch_size
            w_hid -= learning_rate * grad_w_hid / batch_size

        losses.append(epoch_loss / len(X))

    return w_hid, w_out, losses

def predict(X, w_hid, w_out):
    pred_output = []
    for index, arr in X.iterrows():
        arr = np.array(arr).reshape(1, -1)
        hid_output = sigmoid(np.dot(arr, w_hid))
        output = relu(np.dot(hid_output, w_out))
        pred_output.append(np.argmax(output) + 1)

    return pred_output

def calculate_accuracy(predicted, actual):
    correct = sum(p == a for p, a in zip(predicted, actual))
    total = len(actual)
    accuracy = correct / total
    return accuracy

data = pd.read_csv("wine.data", header=None)
Y = data.iloc[:, 0]
X = data.iloc[:, 1:]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

in_units = X_train.shape[1]
hid_units = 64
out_units = len(np.unique(Y))
batch_size = 4

learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001]
colors = ['blue', 'green', 'red', 'orange', 'purple']  # Specify colors for each learning rate
accuracies = []
losses_list = []

for learning_rate in learning_rates:
    print(f"Training with learning rate: {learning_rate}")

    w_hid, w_out, losses = train(X_train, Y_train, in_units, hid_units, out_units, learning_rate, batch_size)
    pred_output = predict(X_test, w_hid, w_out)
    accuracy = calculate_accuracy(pred_output, Y_test.values)
    accuracies.append(accuracy)
    losses_list.append(losses)
    print(f'Accuracy on test set: {accuracy:.4f}')

    print("Classification Report:")
    print(classification_report(Y_test, pred_output, zero_division="warn"))

plt.figure(figsize=(12, 6))
# Plot Accuracy
plt.subplot(1, 2, 1)
for lr, acc, color in zip(learning_rates, accuracies, colors):
    plt.plot(lr, acc, marker='o', color=color, label=f'LR: {lr}')
plt.title('Learning Rate vs Accuracy')
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy')
plt.xscale('log')
plt.grid(True)
plt.legend()

# Plot Loss
plt.subplot(1, 2, 2)
for lr, losses, color in zip(learning_rates, losses_list, colors):
    plt.plot(range(len(losses)), losses, color=color, label=f'LR: {lr}')
plt.title('Epochs vs Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
