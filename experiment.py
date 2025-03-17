import os.path
import argparse
from data.dataset import IOBinClassificationDataSet
from models.ionet.dense_dnn import IONETDenseDNN, ModelA, ModelB, ModelC, ModelD
from models.ionet.logistic_regression import IONETLogisticRegression
from models.ionet.decision_tree import IONETDecisionTree
from models.ionet.random_forest import IONETRandomForest

parser = argparse.ArgumentParser(description='training')
parser.add_argument('-i', '--input', metavar='input',
                    required=True, dest='input',
                    help='Data folder.')
args = parser.parse_args()

data_path = args.input
for osd_idx in range(4):
    path = os.path.join(data_path, f'osd{osd_idx}')
    train_dataset = IOBinClassificationDataSet(path, stage='train')
    test_dataset = IOBinClassificationDataSet(path, stage='test')

    try:
        model = IONETLogisticRegression(path)
        model.train_dataset = train_dataset
        model.test_dataset = test_dataset
        model.train()
        _, report = model.test()
    except Exception as e:
        report = e
    with open(f'osd{osd_idx}_regression.txt', 'w') as file:
        file.write(report)

    try:
        model = IONETRandomForest(path)
        model.train_dataset = train_dataset
        model.test_dataset = test_dataset
        model.train()
        _, report = model.test()
    except Exception as e:
        report = e
    with open(f'osd{osd_idx}_random_forest.txt', 'w') as file:
        file.write(report)

    try:
        model = IONETDecisionTree(path)
        model.train_dataset = train_dataset
        model.test_dataset = test_dataset
        model.train()
        _, report = model.test()
    except Exception as e:
        report = e
    with open(f'osd{osd_idx}_dec_tree.txt', 'w') as file:
        file.write(report)

    for model_class in [ModelA, ModelB, ModelC, ModelD]:
        with open(f'osd{osd_idx}_dnn_{model_class}.txt', 'w') as file:
            try:
                model = IONETDenseDNN(path, model_class=model_class, output=file)
                model.train()
            except Exception as e:
                file.write(e)
