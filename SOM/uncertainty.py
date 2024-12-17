from ast import literal_eval

from chemprop.args import TrainArgs, PredictArgs
from chemprop.data import get_task_names, get_data, MoleculeDataLoader
from chemprop.models import MoleculeModel
from chemprop.uncertainty import UncertaintyEstimator
from chemprop.train.make_predictions import predict_and_save
from chemprop.train import predict
from chemprop.multitask_utils import reshape_values
from tqdm import tqdm
from chemprop.utils import load_checkpoint
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np


def get_enzyme_data(data_path, enzyme_col, enzyme_flag='atom', descriptor_col=[], smi_col='Smiles'):
    dataframe = pd.read_csv(data_path, index_col=False)
    atom_cols = ['atom_{}'.format(str(i)) for i in enzyme_col]
    bond_cols = ['bond_{}'.format(str(i)) for i in enzyme_col]

    enzyme_index = []
    for idx in dataframe.index:
        atom_onehot_list = [literal_eval(i) for i in dataframe.loc[[idx], atom_cols].values[0]]
        bond_onehot_list = [literal_eval(i) for i in dataframe.loc[[idx], bond_cols].values[0]]
        if enzyme_flag == 'atom':
            if np.sum(atom_onehot_list) != 0:
                enzyme_index.append(idx)
        elif enzyme_flag == 'bond':
            if np.sum(bond_onehot_list) != 0:
                enzyme_index.append(idx)

    col = [i.split(',') for i in ['atom_{},bond_{}'.format(str(i), str(i)) for i in enzyme_col]]
    col = sum(col, [])
    col = [smi_col] + col + descriptor_col

    col = list(dataframe.columns)
    new_dataframe = dataframe.loc[enzyme_index, col]
    new_dataframe.to_csv('./Uncertainty/select_col.csv', index=False)
    return './Uncertainty/select_col.csv'


def cal_Youden_index(y_true, y_pred):
    threshold = 0.5  # Define a threshold for binarization

    # Binarize continuous predictions based on the threshold
    # y_pred = (y_pred >= threshold).astype(int)
    # 计算混淆矩阵
    conf_matrix = confusion_matrix(y_true, y_pred)

    # 提取混淆矩阵的 True Positive（TP）、True Negative（TN）、False Positive（FP）、False Negative（FN）
    TP = conf_matrix[1, 1]
    TN = conf_matrix[0, 0]
    FP = conf_matrix[0, 1]
    FN = conf_matrix[1, 0]

    # 计算敏感性（True Positive Rate）和特异性（True Negative Rate）
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)

    # 计算Youden指数
    Youden_index = sensitivity + specificity - 1
    return Youden_index

def cal_uncla(model_list, test_datapath, enzyme_col=[0], enzyme_type='atom'):
    models = [load_checkpoint(checkpoint_path) for checkpoint_path in model_list]
    num_models = len(model_list)
    testpath = get_enzyme_data(test_datapath, enzyme_col=enzyme_col, enzyme_flag=enzyme_type, descriptor_col=[],
                               smi_col='Smiles')

    # getdata
    arguments = ['--data_path', './UGT_process/cyp_total_multi.csv',
                 '--dataset_type', 'classification',
                 ]
    args = TrainArgs().parse_args(arguments)
    args.data_path = testpath
    #     args.batch_size = 1
    if enzyme_type == 'atom':
        args.bond_targets = []
        args.smiles_columns = ['Smiles']
        args.target_columns = ['atom_{}'.format(str(i)) for i in range(9)]
        args.is_atom_bond_targets = True
        args.atom_targets = ['atom_{}'.format(str(i)) for i in range(9)]
    elif enzyme_type == 'bond':
        args.atom_targets = []
        args.smiles_columns = ['Smiles']
        args.target_columns = ['bond_{}'.format(str(i)) for i in range(9)]
        args.is_atom_bond_targets = True
        args.bond_targets = ['bond_{}'.format(str(i)) for i in range(9)]
    args.task_names = get_task_names(path=args.data_path, smiles_columns=args.smiles_columns,
                                     target_columns=args.target_columns, ignore_columns=args.ignore_columns)
    test_data = get_data(path=args.data_path,
                         args=args,
                         skip_none_targets=True)
    test_data_loader = MoleculeDataLoader(dataset=test_data, batch_size=1,
                                          num_workers=0)

    for i, model in enumerate(models):
        preds = predict(
            model=model,
            data_loader=test_data_loader,
            scaler=None,
            atom_bond_scaler=None,
            return_unc_parameters=False,
        )
        if i == 0:
            sum_preds = np.array(preds)
            sum_squared = np.square(preds)
        else:
            sum_preds += np.array(preds)
            sum_squared += np.square(preds)

    if model.is_atom_bond_targets:
        num_tasks = len(sum_preds)
        uncal_preds, uncal_vars = [], []
        for pred, squared in zip(sum_preds, sum_squared):
            uncal_pred = pred / num_models
            uncal_var = (
                    squared / num_models - np.square(pred) / num_models ** 2
            )
            uncal_preds.append(uncal_pred)
            uncal_vars.append(uncal_var)
        uncal_preds_0 = reshape_values(
            uncal_preds,
            test_data,
            len(model.atom_targets),
            len(model.bond_targets),
            num_tasks,
        )
        uncal_vars_0 = reshape_values(
            uncal_vars,
            test_data,
            len(model.atom_targets),
            len(model.bond_targets),
            num_tasks,
        )
    return uncal_preds_0, uncal_vars_0, testpath

def get_yudendix(uncal_preds, uncal_vars, testpath, true_col='atom_0'):
    test_pd = pd.read_csv(testpath, index_col=False)

    mole_num = uncal_preds.shape[0]
    pred = [uncal_preds[i][0] for i in range(mole_num)]
    uncla = [uncal_vars[i][0] for i in range(mole_num)]
    pred_binary = [[int(s >= 0.5) for s in i] for i in pred]
    label_true = [literal_eval(i) for i in test_pd[true_col]]
    uncla_pd = pd.DataFrame({'y_true': label_true, 'y_pred': pred, 'y_pred_label': pred_binary, 'uncal': uncla})
    Correct_type = []
    for i in range(len(uncla_pd)):
        y_pred = uncla_pd['y_pred_label'][i]
        y_true = uncla_pd['y_true'][i]
        if len(y_pred) == len(y_true):
            correct = []
            for num in range(len(y_pred)):
                if y_pred[num] == y_true[num]:
                    correct.append(1)
                else:
                    correct.append(0)
            Correct_type.append(correct)
        else:
            print(i, len(y_true), len(y_pred))

    uncla_pd['Correct'] = Correct_type
    uncla_all = sum([i.tolist() for i in list(uncla_pd['uncal'])], [])
    correct_all = sum([i for i in list(uncla_pd['Correct'])], [])
    Youden_index_list = []
    for i in range(len(uncla_all)):
        confidence = [int(unclas <= uncla_all[i]) for unclas in uncla_all]
        Youden_index = cal_Youden_index(correct_all, confidence)
        Youden_index_list.append(Youden_index)
        if Youden_index == max(Youden_index_list):
            uncar_threshold = uncla_all[i]
    return max(Youden_index_list), uncar_threshold, uncla_pd


if __name__ == '__main__':

    # './Uncertainty/a1/fold_3/test.csv'
    data_dic = {'max_yuden_index': [], 'uncar_threshold': []}

    for i in range(9):
        model_list = ['./Uncertainty/seed_yuan_type19/a{}/fold_0/model_0_{}/model.pt'.format(str(i), 'bond') for i in
                      range(1, 11)]
        test_datapath = './Uncertainty/a1/fold_3/test.csv'
        w = cal_uncla(model_list, test_datapath, enzyme_col=[i], enzyme_type='bond')
        max_yuden_index, uncar_threshold, pd0 = get_yudendix(w[0], w[1], w[2], true_col='bond_{}'.format(str(i)))
        data_dic['max_yuden_index'].append(max_yuden_index)
        data_dic['uncar_threshold'].append(uncar_threshold)
        print('enzyme{}'.format(str(i)))
        print(f'max_yuden_index  :{max_yuden_index}')
        print(f'uncar_threshold  :{uncar_threshold}')
