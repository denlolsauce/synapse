#Rian O Donnell 2024
import torch
from torch.nn import (ReLU, MaxPool1d, Conv1d, BatchNorm1d, Linear, Dropout, LeakyReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout, LogSoftmax, Flatten)
from torch.optim import SGD
import pandas as pd
import csv
import numpy as np
# Read the CSV file
test_data = pd.read_csv("train.csv")
array = []
# View the first 5 rows
print(test_data.head())
array.append(test_data.head())
print(array)
labels = test_data['label'].tolist()
print(labels)
file = open('train.csv')
type(file)
csvreader = csv.reader(file)
rows = []
count = 0
for row in csvreader:

    count += 1
    if count == 1:
        print("hi")
    else:
        print(count)
        row.pop(0)


        rows.append(row)
        if count == 20000:
            break

for i in labels:
    if i == -1:
        print("hi")
        idx = labels.index(i)
        if idx <= 20000:
            rows.pop(idx)
            labels.pop(idx)

        else:
            break

print(rows[0])
print(labels[0])
'''
rows = np.array([rows], dtype=np.float32)
print(rows.shape)
rows = rows.reshape(12885, 1, 14, 256)
print(rows.shape)
print(rows[:2])

print(rows.shape)
'''
rows = np.array(rows, dtype=np.float32)
shape = rows.shape
shape = shape[0]

rows = rows.reshape(shape, 1, 3584)



class Net(Module):
    def __init__(self):
        super(Net, self).__init__()
        self.cnn_layers = Sequential(
            Conv1d(1,14, kernel_size=5, padding=1),
            BatchNorm1d(14),
            LeakyReLU(0.1),
            MaxPool1d(kernel_size=5, stride=1),

        )
        self.cnn_layer2 = Sequential(
            Conv1d(14, 10,kernel_size=5, padding=1),
            BatchNorm1d(10),
            LeakyReLU(0.1),
            MaxPool1d(kernel_size=5, stride=1),
            Dropout(0.2),
        )
        self.cnn_layer3 = Sequential(
            Conv1d(10, 10, kernel_size=5, padding=1),
            BatchNorm1d(10),
            LeakyReLU(0.1),
            MaxPool1d(kernel_size=5, stride=1),
            Dropout(0.2),
        )
        self.linear_layer1 = Sequential(
            Linear(in_features=35660,out_features=3566),
            BatchNorm1d(3566),
            LeakyReLU(0.1),
            Dropout(0.2)

        )
        self.linear_layer2 = Sequential(
            Linear(in_features=3566,out_features=10),

        )
        self.linear_layer3 = Sequential(
            Linear(in_features=250, out_features=10),
        )
        self.linear_layer4 = Sequential(
            Linear(in_features=250, out_features=10)
        )
        self.logsoft = Sequential(
            LogSoftmax(dim=1)
        )
        self.flatten = Sequential(
            Flatten() # probably has to be changed
        )

    def forward(self, x):
        x = self.cnn_layers(x)

        x = self.cnn_layer2(x)

        x = self.cnn_layer3(x)

        x = self.flatten(x)
        x = self.linear_layer1(x)
        x = self.linear_layer2(x)


        return x



model = Net()

#Build a dataset by taking 255 columns and grouping them into 14 * 255 channels
class CustomDataSet():
    def __init__(self, csv_file, label,  transform=None):
        self.df = csv_file
        self.transform = transform
        self.label = label

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        scan = (self.df[index])
        label = self.label[index]
        if self.transform:
            scan = self.transform(scan)

        return scan, label

train_dataset = rows
print(train_dataset.shape)

train_dataset = CustomDataSet(csv_file=rows, label=(labels))

optimizer = SGD(model.parameters(), lr=0.001, weight_decay=5.0e-5)
criterion = CrossEntropyLoss()
num_epochs = 500
train_loss_list = []
batch_size = 128
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}:', end=' ')
    train_loss = 0

    # Iterating over the training dataset in batches
    total_correct = 0
    total_samples = 0
    model.train()
    for i, (scan, labels) in enumerate(train_loader):
        # Extracting images and target labels for the batch being iterated



        # Calculating the model output and the cross entropy loss

        outputs = model(scan)
        print(outputs.shape)

        loss = criterion(outputs, labels)

        # Updating weights according to calculated loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)

        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)
        # Printing loss for each epoch
    accuracy = 100 * total_correct / total_samples
    print("Accuracy: ", accuracy)
    train_loss_list.append(train_loss / len(train_loader))
    print(f"Training loss = {train_loss_list[-1]}")
torch.save(model.state_dict(), 'C:\Desktop\MNISTeeg')
# Plotting loss for all epochs
