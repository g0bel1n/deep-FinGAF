import torch
import numpy as np
from models import CNN
from DataLoad import ImageDataset
from torch.utils.data import DataLoader

dataset = ImageDataset("../data/TEST/labels.txt", "../data/TEST")


def hm_accuracy(ytrue, ypred):
    score=0
    for true_value,pred_value in zip(ytrue,ypred):
        if true_value!=pred_value: score+=1
    return score/len(ytrue)

def translator(pred):
    return 0 if pred[0]>pred[1] else 1


test_loader = DataLoader(dataset, shuffle=True, batch_size=1)

def test_model(model_path: str):
    model =CNN()
    model.load_state_dict(torch.load(model_path))

    test_preds, test_true=[],[]
    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch
            preds = list(np.ravel(model(images/255.).numpy()))
            test_preds.append(preds)
            test_true.append(labels.numpy()[0])
    L = [translator(el) for el in test_preds]

    print(hm_accuracy(test_true, L))

if __name__ =="__main__":
    test_model("../saved_model/cnn")
