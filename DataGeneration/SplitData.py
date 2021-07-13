import os
from random import shuffle

def split_data(test_split, path = "../data"):

    listdir = [el for el in os.listdir(path) if el.endswith(".jpg") and not el.startswith(".")]
    shuffle(listdir)
    ind_split = int(len(listdir)*test_split)
    test_list, train_list  = listdir[:ind_split], listdir[ind_split:]


    for fold_name, dataset in [["TRAIN", train_list], ["TEST",test_list]]:
        main_new_path = os.path.join(path, fold_name)
        os.makedirs(main_new_path, exist_ok=True)
        os.replace(os.path.join(path, fold_name+"_labels.txt"), os.path.join(main_new_path,"labels.txt"))
        for img in dataset:
            new_path = os.path.join(main_new_path,img)
            os.replace(os.path.join(path,img), new_path)

if __name__ == "__main__":

    split_data(0.3)




