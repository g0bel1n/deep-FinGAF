import os

import torch

from DataLoad import ImageDataset
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
import torch.optim as optim


def train(model, epochs=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    dataset = ImageDataset("data/TRAIN/labels.txt", "data/TRAIN")

    data_loader = DataLoader(dataset, shuffle=True, batch_size=5)

    for epoch in tqdm(range(epochs)):
        running_loss = 0.
        for i, data in enumerate(data_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs / 255)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if running_loss < 10e-4 :  break
            # print statistics
            running_loss += loss.item()
            if i % 20 == 19:  # print every 20 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.

    print('Finished Training')

    os.makedirs("saved_model", exist_ok=True)

    PATH = 'saved_model/cnn'
    torch.save(model.state_dict(), PATH)


if __name__ == "__main__":
    from models import CNN

    cnn = CNN()

    train(cnn, 50)
