from rdkit import Chem
from sklearn.metrics import roc_auc_score, precision_recall_curve
from collections import defaultdict
from ast import literal_eval
import numpy as np
import torch
from sklearn.utils import check_consistent_length, column_or_1d
import pandas as pd


def auc2(x, y):
    check_consistent_length(x, y)
    x = column_or_1d(x)
    y = column_or_1d(y)

    if x.shape[0] < 2:
        raise ValueError('At least 2 points are needed to compute'
                         ' area under curve, but x.shape = %s' % x.shape)

    direction = 1
    dx = np.diff(x)
    if np.any(dx < 0):
        if np.all(dx <= 0):
            direction = -1
        else:
            raise ValueError("x is neither increasing nor decreasing "
                             ": {}.".format(x))

    area = direction * np.trapz(y, x)
    if isinstance(area, np.memmap):
        # Reductions such as .sum used internally in np.trapz do not return a
        # scalar by default for numpy.memmap instances contrary to
        # regular numpy.ndarray instances.
        area = area.dtype.type(area)
    return area

def auc(pre,label):
    auc = roc_auc_score(label, pre)
    return auc
def prc_auc(pre, label) -> float:
    precision, recall, _ = precision_recall_curve(label, pre)
    return auc2(recall, precision)
def top_k(pre, label):
    pre = torch.tensor(pre)
    value, idx = torch.topk(pre, k=2, dim=0, largest = True)
    top_1 = 0; top_2 = 0; top_3 = 0
    if label[idx[0]] == 1:
        top_1 += 1; top_2 += 1; top_3 += 1
    elif label[idx[1]] == 1:
        top_2 += 1; top_3 += 1
    # elif label[idx[2]] == 1:
    #     top_3 += 1
    return top_1, top_2, top_3
def pre(tpfn):
    precision = tpfn[0] / (tpfn[0] + tpfn[2] + 0.0000001)
    return precision
def recall(tpfn):
    recall_scores = tpfn[0] / (tpfn[0] +tpfn[3] + 0.0000001)
    return recall_scores
def specificity(tpfn):
    spec = tpfn[1]/(tpfn[1] + tpfn[2] + 0.0000001)
    return spec
def accuracy(tpfn):
    acc = (tpfn[0] + tpfn[1])/(tpfn[0] + tpfn[1] + tpfn[2] +tpfn[3] + 0.0000001)
    return acc
def jaccard(tpfn):
    jaccard_score = tpfn[0]/(tpfn[0] + tpfn[2] + tpfn[3] + 0.0000001)
    return jaccard_score



def TPFN(pre, label, threshold=0.5):
    if len(pre) == len(label):
        length = len(pre)
        pre_label = [1 if i >= threshold else 0 for i in pre]
        TP = 0; TN = 0; FP = 0; FN = 0
        for i in range(length):
            if pre_label[i] == label[i]:
                if pre_label[i] == 1:
                    TP += 1
                else:
                    TN += 1
            else:
                if label[i] == 1:
                    FN += 1
                else:
                    FP += 1
    else:
        TP = 0; TN = 0; FP = 0; FN = 0
        print('len(pre)!=len(label)')

    return TP, TN, FP, FN


def metrics_atom(test_pd, pred_pd, col):
    dic_R = defaultdict(list)
    dic_A = defaultdict(list)
    for task in col:
        target = test_pd[task]
        pred = pred_pd[task]
        # A
        target_A = []
        pred_A = []
        for index, mol in enumerate(target):
            mol_target = literal_eval(mol)
            mol_pred = literal_eval(pred[index])
            if sum(mol_target) != 0:
                target_A.extend(mol_target)
                pred_A.extend(mol_pred)
        tpfn_a = TPFN(pre=pred_A, label=target_A)
        dic_A['auc'].append(auc(pred_A, target_A))
        dic_A['acc'].append(accuracy(tpfn_a))
        dic_A['pre'].append(pre(tpfn_a))
        dic_A['recall'].append(recall(tpfn_a))
        dic_A['spec'].append(specificity(tpfn_a))
        dic_A['jaccard'].append(jaccard(tpfn_a))
        # R
        auc_r = []
        acc_r = []
        pre_r = []
        recall_r = []
        spec_r = []
        jaccard_r = []
        top_1 = []
        top_2 = []
        top_3 = []
        for index, mol in enumerate(target):
            mol_target = literal_eval(mol)
            mol_pred = literal_eval(pred[index])
            if sum(mol_target) != 0:
                tpfn_r = TPFN(pre=mol_pred, label=mol_target)
                auc_r.append(auc(mol_pred, mol_target))
                acc_r.append(accuracy(tpfn_r))
                pre_r.append(pre(tpfn_r))
                recall_r.append(recall(tpfn_r))
                spec_r.append(specificity(tpfn_r))
                jaccard_r.append(jaccard(tpfn_r))
                top_1.append(top_k(mol_pred, mol_target)[0])
                top_2.append(top_k(mol_pred, mol_target)[1])
                top_3.append(top_k(mol_pred, mol_target)[2])
        dic_R['auc'].append(np.mean(auc_r))
        dic_R['acc'].append(np.mean(acc_r))
        dic_R['pre'].append(np.mean(pre_r))
        dic_R['recall'].append(np.mean(recall_r))
        dic_R['spec'].append(np.mean(spec_r))
        dic_R['jaccard'].append(np.mean(jaccard_r))
        dic_R['top_1'].append(np.mean(top_1))
        dic_R['top_2'].append(np.mean(top_2))
        dic_R['top_3'].append(np.mean(top_3))

    return dic_A, dic_R


def metrics_fix(test_pd, pred_atom_pd, pred_bond_pd, col_range):
    dic_R = defaultdict(list)
    dic_A = defaultdict(list)
    for task_num in col_range:
        if task_num != 'ugt':
            target_atom = test_pd['atom_{}'.format(str(task_num))]
            target_bond = test_pd['bond_{}'.format(str(task_num))]
            pred_atom = pred_atom_pd['atom_{}'.format(str(task_num))]
            pred_bond = pred_bond_pd['bond_{}'.format(str(task_num))]
        # A
        target_A = []
        pred_A = []
        for index in range(len(target_atom)):
            mol_target_atom = literal_eval(target_atom[index])
            mol_target_bond = literal_eval(target_bond[index])
            mol_pred_atom = literal_eval(pred_atom[index])
            mol_pred_bond = literal_eval(pred_bond[index])
            if sum(mol_target_atom) != 0:
                target_A.extend(mol_target_atom)
                pred_A.extend(mol_pred_atom)
            if sum(mol_target_bond) != 0:
                target_A.extend(mol_target_bond)
                pred_A.extend(mol_pred_bond)
        tpfn_a = TPFN(pre=pred_A, label=target_A)
        dic_A['auc'].append(auc(pred_A, target_A))
        dic_A['acc'].append(accuracy(tpfn_a))
        dic_A['pre'].append(pre(tpfn_a))
        dic_A['recall'].append(recall(tpfn_a))
        dic_A['spec'].append(specificity(tpfn_a))
        dic_A['jaccard'].append(jaccard(tpfn_a))
        dic_A['prc'].append(prc_auc(pred_A, target_A))

        # R
        auc_r = []
        acc_r = []
        pre_r = []
        recall_r = []
        spec_r = []
        jaccard_r = []
        prc_r = []
        top_1 = []
        top_2 = []
        top_3 = []
        for index in range(len(target_atom)):
            mol_target_atom = literal_eval(target_atom[index])
            mol_target_bond = literal_eval(target_bond[index])
            mol_pred_atom = literal_eval(pred_atom[index])
            mol_pred_bond = literal_eval(pred_bond[index])
            if sum(mol_target_atom) != 0 and sum(mol_pred_bond) != 0:
                tpfn_r = TPFN(pre=mol_pred_atom + mol_pred_bond, label=mol_target_atom + mol_target_bond)
                try:
                    auc_r.append(auc(mol_pred_atom + mol_pred_bond, mol_target_atom + mol_target_bond))
                    acc_r.append(accuracy(tpfn_r))
                    pre_r.append(pre(tpfn_r))
                    recall_r.append(recall(tpfn_r))
                    spec_r.append(specificity(tpfn_r))
                    jaccard_r.append(jaccard(tpfn_r))
                    prc_r.append(prc_auc(mol_pred_atom + mol_pred_bond, mol_target_atom + mol_target_bond))
                    top_1.append(top_k(mol_pred_atom + mol_pred_bond, mol_target_atom + mol_target_bond)[0])
                    top_2.append(top_k(mol_pred_atom + mol_pred_bond, mol_target_atom + mol_target_bond)[1])
                    top_3.append(top_k(mol_pred_atom + mol_pred_bond, mol_target_atom + mol_target_bond)[2])
                except:
                    pass
            elif sum(mol_target_atom) != 0:
                try:
                    tpfn_r = TPFN(pre=mol_pred_atom, label=mol_target_atom)
                    auc_r.append(auc(mol_pred_atom, mol_target_atom))
                    acc_r.append(accuracy(tpfn_r))
                    pre_r.append(pre(tpfn_r))
                    recall_r.append(recall(tpfn_r))
                    spec_r.append(specificity(tpfn_r))
                    jaccard_r.append(jaccard(tpfn_r))
                    prc_r.append(prc_auc(mol_pred_atom, mol_target_atom))
                    top_1.append(top_k(mol_pred_atom, mol_target_atom)[0])
                    top_2.append(top_k(mol_pred_atom, mol_target_atom)[1])
                    top_3.append(top_k(mol_pred_atom, mol_target_atom)[2])
                except:
                    pass
            elif sum(mol_target_bond) != 0:
                try:
                    tpfn_r = TPFN(pre=mol_pred_bond, label=mol_target_bond)
                    auc_r.append(auc(mol_pred_bond, mol_target_bond))
                    acc_r.append(accuracy(tpfn_r))
                    pre_r.append(pre(tpfn_r))
                    recall_r.append(recall(tpfn_r))
                    spec_r.append(specificity(tpfn_r))
                    jaccard_r.append(jaccard(tpfn_r))
                    prc_r.append(prc_auc(mol_pred_bond, mol_target_bond))
                    top_1.append(top_k(mol_pred_bond, mol_target_bond)[0])
                    top_2.append(top_k(mol_pred_bond, mol_target_bond)[1])
                    top_3.append(top_k(mol_pred_bond, mol_target_bond)[2])
                except:
                    pass
        dic_R['auc'].append(np.mean(auc_r))
        dic_R['acc'].append(np.mean(acc_r))
        dic_R['pre'].append(np.mean(pre_r))
        dic_R['recall'].append(np.mean(recall_r))
        dic_R['spec'].append(np.mean(spec_r))
        dic_R['jaccard'].append(np.mean(jaccard_r))
        dic_R['prc'].append(np.mean(prc_r))
        dic_R['top_1'].append(np.mean(top_1))
        dic_R['top_2'].append(np.mean(top_2))
        dic_R['top_3'].append(np.mean(top_3))
    return dic_A, dic_R


def cal_metrics(test, test_pred, threshold=0.5):
    """
    Function to compute various evaluation metrics

    Parameters:
    test : DataFrame
        DataFrame containing the true values.
    test_pred : DataFrame
        DataFrame containing the predicted values.
    threshold : float, optional, default=0.5
        Threshold to convert predicted values into binary classification.

    Returns:
    result_df : DataFrame
        A DataFrame containing various evaluation metrics, with metric names as row indices and feature names as columns.
    """

    dic_result = defaultdict(list)
    cal_cols = test_pred.columns[1:]

    for col in cal_cols:
        nan_list = test[test[col].isna()].index.tolist()
        test_list = test.drop(nan_list)[col].tolist()
        pred_list = test_pred.drop(nan_list)[col].tolist()
        tpfn = TPFN(pred_list, test_list, threshold=threshold)
        dic_result['auc'].append(auc(pred_list, test_list))
        dic_result['prc'].append(prc_auc(pred_list, test_list))
        dic_result['acc'].append(accuracy(tpfn))
        dic_result['pre'].append(pre(tpfn))
        dic_result['recall'].append(recall(tpfn))
        dic_result['spec'].append(specificity(tpfn))
        dic_result['sen'].append(recall(tpfn))
        dic_result['jaccard'].append(jaccard(tpfn))

    # get dataframe
    result_df = pd.DataFrame.from_dict(dic_result, orient='index')
    result_df.columns = cal_cols

    return result_df


def get_pd_mean_var(path_list):
    if type(path_list[0]) == str:
        pd_list = []
        for path in path_list:
            dataframe = pd.read_csv(path, index_col=False)
            pd_list.append(dataframe)
    else:
        pd_list = path_list

    num = len(pd_list)
    pd_index = pd_list[0]['Unnamed: 0']
    pd_col = pd_list[0].columns[1:]

    x = []
    for i in range(0, num):
        pd_list[i].drop('Unnamed: 0', axis=1, inplace=True)
        x.append(pd_list[i].values)
    x = np.array(x)
    mean_value = np.mean(x, axis=0)
    var_value = np.var(x, axis=0)

    new_mean = pd.DataFrame(mean_value)
    new_mean.columns = pd_col
    new_mean.index = pd_index

    new_var = pd.DataFrame(var_value)
    new_var.columns = pd_col
    new_var.index = pd_index

    return new_mean, new_var

def get_pd_mean_var2(path_list):
    if type(path_list[0]) == str:
        pd_list = []
        for path in path_list:
            dataframe = pd.read_csv(path, index_col=False)
            pd_list.append(dataframe)
    else:
        pd_list = path_list

    num = len(pd_list)
    pd_index = pd_list[0][pd_list[0].columns[0]]
    pd_col = pd_list[0].columns[1:]
    drop_col = pd_list[0].columns[0]

    x = []
    for i in range(0, num):
        pd_list[i].drop(drop_col, axis=1, inplace=True)
        x.append(pd_list[i].values)

    x = np.array(x)
    mean_value = np.mean(x, axis=0)
    var_value = np.var(x, axis=0)

    new_mean = pd.DataFrame(mean_value)
    new_mean.columns = pd_col
    new_mean.index = pd_index

    new_var = pd.DataFrame(var_value)
    new_var.columns = pd_col
    new_var.index = pd_index

    return new_mean, new_var


def multi_remove_nan(df):
    #   length = len(df.index)
    atom_labels = ['atom_{}'.format(str(i))for i in range(9)]
    bond_labels = ['bond_{}'.format(str(i))for i in range(9)]
    for idx in df.index:
        smiles = df['Smiles'][idx]
        mol = Chem.MolFromSmiles(smiles)
        atom_num = mol.GetNumAtoms()
        bond_num = mol.GetNumBonds()
        for atom_label in atom_labels:
            if type(df[atom_label][idx]) == float:
                df[atom_label][idx] = [0]*atom_num
        for bond_label in bond_labels:
            if type(df[bond_label][idx]) == float:
                df[bond_label][idx] = [0]*bond_num
    return df

def statistics(dataframe,smi_col='Smiles', target_col=range(9)):
    # statistic total num for 9 enzyme
    dic = defaultdict(int)
    for idx, smi in enumerate(dataframe[smi_col]):
        for i in target_col:
            try:
                atom_sum = sum(literal_eval(list(dataframe['atom_{}'.format(str(i))])[idx]))
                bond_sum = sum(literal_eval(list(dataframe['bond_{}'.format(str(i))])[idx]))
            except:
                atom_sum = sum(list(dataframe['atom_{}'.format(str(i))])[idx])
                bond_sum = sum(list(dataframe['bond_{}'.format(str(i))])[idx])
            if atom_sum != 0 or bond_sum != 0:
                dic[i] += 1
    return dic