from ast import literal_eval

from chemprop.utils import load_checkpoint
from chemprop.data.data import MoleculeDataLoader
from chemprop.train import predict
from rdkit import Chem

import descriptors
import pandas as pd
from chemprop.args import TrainArgs
from chemprop.data import get_data, get_task_names, MoleculeDataset

from .load_model import Original_Atom, Original_Bond, AddAtom_Atom, AddAtom_Bond,\
    AddMol_Atom, AddMol_Bond, AddBoth_Atom, AddBoth_Bond
from .utils import get_pd_mean, get_pd_variance


# threshold = [0.85, 0.92, 0.94, 0.86, 0.90, 0.94, 0.80, 0.85, 0.85]
threshold = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

def total_pred(atom_model_path, bond_model_path, data_path_pkl, args,
               atom_descriptors=False, mol_descriptors=False, normalization=False):
    model_atom = load_checkpoint(path=atom_model_path)
    model_bond = load_checkpoint(path=bond_model_path)

    dsp_col_atom = ['EffectiveAtomPolarizability', 'PartialSigmaCharge',
               'PartialTChargeMMFF94', 'PiElectronegativity', 'SigmaElectronegativity',
               'ProtonAffinityHOSE', 'PartialTChargePEOE']
    dsp_col_mol = ['BCUT', 'TPSA', 'log', 'csp3', 'mol_vt']
    normal_col = ['BCUT', 'TPSA', 'log', 'csp3', 'mol_vt']
    dsp_col = ['EffectiveAtomPolarizability', 'PartialSigmaCharge',
               'PartialTChargeMMFF94', 'PiElectronegativity', 'SigmaElectronegativity',
               'ProtonAffinityHOSE', 'PartialTChargePEOE', 'BCUT', 'TPSA', 'log', 'csp3', 'mol_vt']

    args.atom_descriptors = "descriptor"
    if atom_descriptors and mol_descriptors and normalization:
        args.atom_descriptors_size = 12
        strat_test_set = pd.read_csv(args.data_path)
        strat_test_set = descriptors.normalization(strat_test_set, normal_col)
        test_descriptor = descriptors.to_pkl(dsp_pd=strat_test_set, dsp_col=dsp_col,
                                             savepath=data_path_pkl + '/descriptors.pkl', smi_col='Smiles')
    elif atom_descriptors and mol_descriptors and not normalization:
        args.atom_descriptors_size = 12
        strat_test_set = pd.read_csv(args.data_path)
        test_descriptor = descriptors.to_pkl(dsp_pd=strat_test_set, dsp_col=dsp_col,
                                             savepath=data_path_pkl + '/descriptors.pkl', smi_col='Smiles')
    elif atom_descriptors:
        args.atom_descriptors_size = 7
        strat_test_set = pd.read_csv(args.data_path)
        test_descriptor = descriptors.to_pkl(dsp_pd=strat_test_set, dsp_col=dsp_col_atom,
                                             savepath=data_path_pkl + '/descriptors.pkl', smi_col='Smiles')
    elif mol_descriptors and normalization:
        args.atom_descriptors_size = 5
        strat_test_set = pd.read_csv(args.data_path)
        strat_test_set = descriptors.normalization(strat_test_set, normal_col)
        test_descriptor = descriptors.to_pkl(dsp_pd=strat_test_set, dsp_col=dsp_col_mol,
                                             savepath=data_path_pkl + '/descriptors.pkl', smi_col='Smiles')
    elif mol_descriptors and not normalization:
        args.atom_descriptors_size = 5
        strat_test_set = pd.read_csv(args.data_path)
        test_descriptor = descriptors.to_pkl(dsp_pd=strat_test_set, dsp_col=dsp_col_mol,
                                             savepath=data_path_pkl + '/descriptors.pkl', smi_col='Smiles')
    else:
        strat_test_set = pd.read_csv(args.data_path)

    # atom
    args.batch_size = 1
    args.dropout = 0.1
    args.bond_targets = []
    args.smiles_columns = ['Smiles']
    args.target_columns = ['atom_{}'.format(str(i)) for i in range(9)]
    args.is_atom_bond_targets = True
    args.atom_targets = ['atom_{}'.format(str(i)) for i in range(9)]
    args.task_names = get_task_names(path=args.data_path, smiles_columns=args.smiles_columns,
                                     target_columns=args.target_columns, ignore_columns=args.ignore_columns)
    if mol_descriptors or atom_descriptors:
        data_atom = get_data(path=args.data_path,
                             args=args,
                             atom_descriptors_path=data_path_pkl + '/descriptors.pkl',
                             skip_none_targets=True)
    else:
        args.atom_descriptors = None
        data_atom = get_data(path=args.data_path,
                             args=args,
                             atom_descriptors_path=None,
                             skip_none_targets=True)
    test_data_loader_atom = MoleculeDataLoader(dataset=data_atom, batch_size=1,
                                               num_workers=0)
    test_preds_atom = predict(model=model_atom, data_loader=test_data_loader_atom)
    # bond
    args.batch_size = 1
    args.dropout = 0.0
    args.atom_targets = []
    args.smiles_columns = ['Smiles']
    args.target_columns = ['bond_{}'.format(str(i)) for i in range(9)]
    args.is_atom_bond_targets = True
    args.bond_targets = ['bond_{}'.format(str(i)) for i in range(9)]
    args.task_names = get_task_names(path=args.data_path, smiles_columns=args.smiles_columns,
                                     target_columns=args.target_columns, ignore_columns=args.ignore_columns)
    if atom_descriptors or mol_descriptors:
        data_bond = get_data(path=args.data_path,
                             args=args,
                             atom_descriptors_path=data_path_pkl + '/descriptors.pkl',
                             skip_none_targets=True)
    else:
        args.atom_descriptors = None
        data_bond = get_data(path=args.data_path,
                             args=args,
                             atom_descriptors_path=None,
                             skip_none_targets=True)

    test_data_loader_bond = MoleculeDataLoader(dataset=data_bond, batch_size=1,
                                               num_workers=0)
    test_preds_bond = predict(model=model_bond, data_loader=test_data_loader_bond)

    smis = strat_test_set['Smiles']
    atom_num = []
    bond_num = []
    for smi in smis:
        mol = Chem.MolFromSmiles(smi)
        atom_num.append(mol.GetNumAtoms())
        bond_num.append(mol.GetNumBonds())

    pd_pred = pd.DataFrame()
    pd_pred['Smiles'] = smis

    for i in range(9):
        atom_pred = []
        bond_pred = []
        atom_num_flag = 0
        bond_num_flag = 0
        for idx, atomNum in enumerate(atom_num):
            atom_pred.append(list(test_preds_atom[i].reshape(-1))[atom_num_flag:atom_num_flag + atomNum])
            atom_num_flag += atomNum
            bond_pred.append(list(test_preds_bond[i].reshape(-1))[bond_num_flag:bond_num_flag + bond_num[idx]])
            bond_num_flag += bond_num[idx]
        pd_pred['atom_{}'.format(str(i))] = atom_pred
        pd_pred['bond_{}'.format(str(i))] = bond_pred

    return pd_pred

def get_target_label(df, threshold):

    atom_labels = ['atom_{}'.format(str(i)) for i in range(9)]
    bond_labels = ['bond_{}'.format(str(i)) for i in range(9)]

    df2 = pd.DataFrame()
    df2['Smiles'] = df['Smiles']
    for idx, atom_c in enumerate(atom_labels):
        df2[atom_c] = df[atom_c].apply(lambda x: [1 if i >= threshold[idx] else 0 for i in x])
    for idx, bond_c in enumerate(bond_labels):
        df2[bond_c] = df[bond_c].apply(lambda x: [1 if i >= threshold[idx] else 0 for i in x])

    return df2

atom_std = [0.0006963660030625761, 0.00032402321812696755, 0.0007557964418083429,
            0.0005471919430419803, 0.00045517960097640753, 0.0006426861509680748,
            0.0004963660030625761, 0.00045517960097640753, 0.0006448626518249512]
bond_std = [0.007335423957556486, 0.007393424399197102, 0.02744896709918976,
            0.02745058760046959, 0.00488391425460577, 0.02688596397638321,
            0.02688596397638321, 0.006906951777637005, 0.02684260532259941]

def get_var_label(pd, atom_std=atom_std, bond_std=bond_std):
    atom_labels = ['atom_{}'.format(str(i)) for i in range(9)]
    bond_labels = ['bond_{}'.format(str(i)) for i in range(9)]
    for idx, atom_c in enumerate(atom_labels):
        pd[atom_c] = pd[atom_c].apply(lambda x: [0 if i >= atom_std[idx] else 1 for i in x])
    for idx, bond_c in enumerate(bond_labels):
        pd[bond_c] = pd[bond_c].apply(lambda x: [0 if i >= bond_std[idx] else 1 for i in x])

    return pd

def predict_csv(datapath, savepathpkl, atom_descriptors=False, mol_descriptors=False,
                normalization=False, ensemble=False, threshold=threshold):

    """
    Predicts the target labels for a dataset specified by the datapath, using pre-trained models for atoms and bonds.

    Args:
    datapath (str or pd.DataFrame): The path to the CSV file containing the data or a DataFrame object.
    savepathpkl (str): The path where intermediate files will be saved.
    atom_model_path (str): The path to the pre-trained atom model.
    bond_model_path (str): The path to the pre-trained bond model.
    atom_descriptors (bool, optional): Flag to indicate whether atom descriptors should be used. Default is False.
    mol_descriptors (bool, optional): Flag to indicate whether molecule descriptors should be used. Default is False.
    normalization (bool, optional): Flag to indicate whether normalization should be applied. Default is False.

    Returns:
    pd.DataFrame: A DataFrame containing the predicted labels for the dataset.
    """

    arguments = ['--data_path', './test.csv',
                 '--dataset_type', 'classification']
    args = TrainArgs().parse_args(arguments)

    if type(datapath) == pd.core.frame.DataFrame:
        datapath.to_csv(savepathpkl+'/dataset.csv', index=False)
        args.data_path = savepathpkl+'/dataset.csv'
        data_path_pkl = savepathpkl
    else:
        args.data_path = datapath
        data_path_pkl = savepathpkl

    # 3 types: original; add atom; add mol; add both
    if not ensemble:
        if atom_descriptors and mol_descriptors:
            atom_model_path = AddBoth_Atom[0]
            bond_model_path = AddBoth_Bond[0]
            # atom_model_path = new_Atom[0]
            # bond_model_path = new_Bond[0]
        elif atom_descriptors:
            atom_model_path = AddAtom_Atom[0]
            bond_model_path = AddAtom_Bond[0]
        elif mol_descriptors:
            atom_model_path = AddMol_Atom[0]
            bond_model_path = AddMol_Bond[0]
        else:
            atom_model_path = Original_Atom[2]
            bond_model_path = Original_Bond[2]
        pred = total_pred(atom_model_path=atom_model_path, bond_model_path=bond_model_path,
                          data_path_pkl=data_path_pkl, args=args,
                          atom_descriptors=atom_descriptors, mol_descriptors=mol_descriptors, normalization=normalization)
        pred_label = get_target_label(pred, threshold=threshold)
    else:
        if atom_descriptors and mol_descriptors:
            model_atom_paths = AddBoth_Atom
            model_bond_paths = AddBoth_Bond
        elif atom_descriptors:
            model_atom_paths = AddAtom_Atom
            model_bond_paths = AddAtom_Bond
        elif mol_descriptors:
            model_atom_paths = AddMol_Atom
            model_bond_paths = AddMol_Bond
        else:
            model_atom_paths = Original_Atom
            model_bond_paths = Original_Bond
        pred_list = []
        for i in range(5):
            atom_model_path = model_atom_paths[i]
            bond_model_path = model_bond_paths[i]
            pred = total_pred(atom_model_path=atom_model_path, bond_model_path=bond_model_path,
                              data_path_pkl=data_path_pkl, args=args,
                              atom_descriptors=atom_descriptors, mol_descriptors=mol_descriptors,
                              normalization=normalization)
            # pred.to_csv(f'./okk{str(i)}.csv', index=False)
            pred_list.append(pred)
        pred = get_pd_mean(pd_list=pred_list, drop_c=['Smiles'])
        var = get_pd_variance(pd_list=pred_list, drop_c=['Smiles'])
        pred_label = get_target_label(pred, threshold=threshold)
        var_label = get_var_label(var, atom_std, bond_std)
    if ensemble:
        return pred, pred_label, var_label
    else:
        return pred, pred_label

def predict_single(enzyme_idx,data_path):

    arguments = ['--data_path', './test.csv',
                 '--dataset_type', 'classification']
    args = TrainArgs().parse_args(arguments)
    args.data_path = data_path

    # model
    atom_model = f'./model/single_model/atom_result/result_{enzyme_idx}/model_0/model.pt'
    bond_model = f'./model/single_model/bond_result/result_{enzyme_idx}/model_0/model.pt'
    model_atom = load_checkpoint(path=atom_model)
    model_bond = load_checkpoint(path=bond_model)

    # atom
    args.smiles_columns = ['Smiles']
    args.target_columns = ['atom_{}'.format(str(enzyme_idx))]
    args.is_atom_bond_targets = True
    args.atom_targets = ['atom_{}'.format(str(enzyme_idx))]
    args.task_names = get_task_names(path=args.data_path, smiles_columns=args.smiles_columns,
                                     target_columns=args.target_columns, ignore_columns=args.ignore_columns)

    args.atom_descriptors = None
    data_atom = get_data(path=args.data_path,
                         args=args,
                         atom_descriptors_path=None,
                         skip_none_targets=True)
    test_data_loader_atom = MoleculeDataLoader(dataset=data_atom, batch_size=1,
                                               num_workers=0)
    test_preds_atom = predict(model=model_atom, data_loader=test_data_loader_atom)

    # bond
    args.batch_size = 1
    args.dropout = 0.0
    args.atom_targets = []
    args.smiles_columns = ['Smiles']
    args.target_columns = ['bond_{}'.format(str(i)) for i in range(9)]
    args.is_atom_bond_targets = True
    args.bond_targets = ['bond_{}'.format(str(i)) for i in range(9)]
    args.task_names = get_task_names(path=args.data_path, smiles_columns=args.smiles_columns,
                                     target_columns=args.target_columns, ignore_columns=args.ignore_columns)

    args.atom_descriptors = None
    data_bond = get_data(path=args.data_path,
                         args=args,
                         atom_descriptors_path=None,
                         skip_none_targets=True)

    test_data_loader_bond = MoleculeDataLoader(dataset=data_bond, batch_size=1,
                                               num_workers=0)
    test_preds_bond = predict(model=model_bond, data_loader=test_data_loader_bond)


if __name__ == '__main__':

    pred = predict_csv(datapath='./muti_cdk_eb2_nan.csv', savepathpkl='./testfile',
                       atom_descriptors=False, mol_descriptors=False, normalization=False, ensemble=False)
    # pred[1].to_csv('./testfile/result_plot.csv')
    # pred = predict_single(data_path='./datasets_in/enzyme0.csv',enzyme_idx=0)
    # pred.to_csv('./testfile/result_new.csv', index=False)
    # var.to_csv('./testfile/result_var.csv', index=False)

    # # external
    # pred = predict_csv(datapath='external222.csv', savepathpkl='./testfile',
    #                    atom_descriptors=False, mol_descriptors=False, normalization=False, ensemble=False)
    # pred.to_csv('./script/yuan3_3.csv', index=False)

    # pred, pred_label, var = predict_csv(datapath='./test_pre.csv', savepathpkl='./testfile', atom_descriptors=False,
    #                         mol_descriptors=False, normalization=False, ensemble=True, threshold=threshold)
    # print(pred)
    # print(pred_label)
    # var.to_csv('./try_var_value.csv', index_label=False)

