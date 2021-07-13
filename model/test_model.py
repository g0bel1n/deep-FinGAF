import torch
from models import CNN

def test_model(model_path: str):
    model =CNN()
    model.load_state_dict(torch.load(model_path))



    test_preds=[]
    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch
            preds = model(images/255.).numpy()
            test_preds.append(preds)

    print(test_preds)