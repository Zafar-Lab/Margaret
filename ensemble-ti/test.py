import ast
import os
import click
import logging
import numpy as np
import pandas as pd
import torch

from prettytable import PrettyTable
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm
from models import *
from datasets import *
from datasets.np import NpDataset
from util.config import configure_device
from util.datastore import get_dataset


logging.basicConfig()
logger = logging.getLogger('test')
logger.setLevel(logging.INFO)


LABEL_MAP = ['RIF', 'INH', 'PZA', 'EMB', 'STR', 'CIP', 'CAP', 'AMK', 'MOXI', 'OFLX', 'KAN']


@click.group()
def cli():
    pass


# Run this command to generate the predictions.
# TODO: Complete the template, to output the predictions in a .csv file
# The number of parameters to this command might change based on the evaluation strategy!
@cli.command()
@click.option('-i', '--ignore-index', default=-1)
@click.option('-b', '--batch-size', default=32)
@click.option('-d', '--device', default='gpu')
@click.option('-s', '--save-path', default=os.getcwd())
@click.option('--drugs', default='[]')
@click.option('--model',
                type=click.Choice(['wdnn256', 'wdnn512', 'deepcnn', 'deepcnn_se'], case_sensitive=False), default='deepcnn')
@click.argument('dataset_name')
@click.argument('test_file_path')
@click.argument('base_pred_file_path')
@click.argument('chkpt_path')
def predict(dataset_name, test_file_path, base_pred_file_path, chkpt_path,
    save_path=os.getcwd(), device='gpu', batch_size=32, ignore_index=-1, drugs='[]', model='deepcnn'):
    """Generates predictions for the test dataset. \n
    Sample command: \n
    python test.py predict mutation \
            MTB/data/X_test_1.csv\
            MTB/labels/Y_testData_1_nolabels_RIF.csv\
            MTB/checkpoints/chkpt_mutation_Adam_e49_rif_deepcnn.pt\
            --model deepcnn --drugs '["RIF"]'
    """
    drugs = ast.literal_eval(drugs)
    drugs = LABEL_MAP if drugs == [] else drugs
    name2id = {v:k for k, v in enumerate(drugs)}
    device = configure_device(device)

    # Read test dataset
    test_data_csv = pd.read_csv(test_file_path)

    # Drop mutation columns not considered during training
    test_data_csv = test_data_csv.drop(columns=['SNP_CN_2714366_C967A_V323L_eis', 'SNP_I_2713795_C329T_inter_Rv2415c_eis', 'SNP_I_2713872_C252A_inter_Rv2415c_eis'])
    test_data_np = test_data_csv.to_numpy(np.float)

    # Read test prediction file
    pred_csv = pd.read_csv(base_pred_file_path)
    # Subtract 1 from the indices as the indices in the test prediction file start from 1
    valid_inds = pred_csv['ID'] - 1
    drug_name = pred_csv.columns[-1]
    drug_id = name2id[drug_name]
    
    # Get the test data on which to make predictions
    valid_test_data = test_data_np[valid_inds]
    test_dataset = NpDataset(valid_test_data)

    # Load model
    model_state = torch.load(chkpt_path)['model']
    num_classes = len(drugs)
    # Select the model
    if model == 'deepcnn':
        net = DeepCNNResistancePredictor(test_dataset.shape[1], num_classes)
    elif model == 'deepcnn_se':
        net = DeepCNNResistancePredictor_se(test_dataset.shape[1], num_classes)
    elif model == 'wdnn256':
        net = WDNNResistancePredictor(test_dataset.shape[1], num_classes)
    elif model == 'wdnn512':
        net = WDNNResistancePredictor(test_dataset.shape[1], num_classes, nodes=[512, 512, 512])
    net.load_state_dict(torch.load(chkpt_path)['model'])
    net.to(device)
    net.eval()
    with torch.no_grad():
        for idx, data in enumerate(test_dataset):
            data = data.to(device).unsqueeze(0)
            pred = torch.sigmoid(net(data)).squeeze(0).cpu().numpy()
            pred_csv.iat[idx, 1] = pred[drug_id]
    file_path = os.path.join(save_path, f'pred_{drug_name}.csv')
    pred_csv.to_csv(file_path, index=False)
    click.echo(f'Generated predictions for drug:{drug_name} at {file_path}')


@cli.command()
@click.option('-i', '--ignore-index', default=-1)
@click.option('-b', '--batch-size', default=32)
@click.option('-d', '--device', default='gpu')
@click.option('--drugs', default='[]')
@click.option('--model',
                type=click.Choice(['wdnn256', 'wdnn512', 'deepcnn', 'deepcnn_se'], case_sensitive=False), default='deepcnn')
@click.argument('dataset_name')
@click.argument('base_dir')
@click.argument('chkpt_path')
def evaluate(dataset_name, base_dir, chkpt_path, drugs='[]', ignore_index=-1, batch_size=32, device='gpu', num_classes=1, model='deepcnn'):
    """Performs evaluation on the Validation dataset\n
    Sample Command:\n
    python test.py evaluate --model deepcnn mutation\
                data/subset\
                checkpoints/chkpt_mutation_Adam_e49_rif_deepcnn.pt\
                --drugs '["RIF"]'

    Args:
        dataset_name ([type]): [Dataset name]
        base_dir ([type]): [Dir where validation files are located]
        chkpt_path ([type]): [Checkpoint path to load]
        ignore_index (int, optional): [Index denoting missing prediction values]. Defaults to -1.
        batch_size (int, optional): [Batch size when evaluating]. Defaults to 32.
        device (str, optional): [Device on which to run the evaluation. TPU not supported yet!]. Defaults to 'gpu'.
    """
    drugs = ast.literal_eval(drugs)
    drugs = LABEL_MAP if drugs == [] else drugs
    device = configure_device(device)
    # Evaluates the model performance on the validation set for neural net models only!
    # Load the dataset
    val_dataset, num_classes = get_dataset(dataset_name, base_dir, mode='val', drugs=drugs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=False, pin_memory=True)

    # Load the model for Multi drug prediction
    if model == 'deepcnn':
        net = DeepCNNResistancePredictor(val_dataset.shape[1], num_classes)
    elif model == 'deepcnn_se':
        net = DeepCNNResistancePredictor_se(val_dataset.shape[1], num_classes)
    elif model == 'wdnn256':
        net = WDNNResistancePredictor(val_dataset.shape[1], num_classes)
    elif model == 'wdnn512':
        net = WDNNResistancePredictor(val_dataset.shape[1], num_classes, nodes=[512, 512, 512])
    net.load_state_dict(torch.load(chkpt_path)['model'])
    net.to(device)
    net.eval()
    with torch.no_grad():
        targets = []
        preds = []
        for data_batch, target_batch in val_loader:
            data_batch = data_batch.to(device)
            val_pred_batch = torch.sigmoid(net(data_batch))
            targets.append(target_batch)
            preds.append(val_pred_batch)
        targets = torch.cat(targets, dim=0).cpu().numpy()
        preds = torch.cat(preds, dim=0).cpu().numpy()

    # num_drugs = targets.shape[-1]
    table = PrettyTable()
    table.field_names = ['Drug Name', 'AUCROC Score', 'Sensitivity', 'Specificity', 'Threshold']
    for drug_idx, drug in enumerate(drugs):
        drug_targets = targets[:, drug_idx]
        drug_preds = preds[:, drug_idx]
        targets_ = drug_targets[drug_targets != ignore_index]
        preds_ = drug_preds[drug_targets != ignore_index]
        if drug == 'CIP':
            # Only one labeled example is present so skip
            continue

        # Compute the sensitivity and specificity
        fpr, tpr, thresholds = roc_curve(targets_, preds_)
        spec_sens_sum = (1 - fpr) + tpr
        best_sum = np.max(spec_sens_sum)
        best_num_index = np.argmax(spec_sens_sum)
        best_specificity = round((1 - fpr[best_num_index]) * 100, 1)
        best_sensitivity = round(tpr[best_num_index] * 100, 1)
        best_threshold = round(thresholds[best_num_index], 2)
        # Compute the ROC AUC metric
        auc_score = round(roc_auc_score(targets_, preds_) * 100, 1)
        table.add_row([drug, auc_score, best_sensitivity, best_specificity, best_threshold])
    click.echo(table)


if __name__ == '__main__':
    cli()
