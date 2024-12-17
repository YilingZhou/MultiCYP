from ast import literal_eval

import numpy as np
import rdkit
from rdkit import Chem
import pandas as pd

def mol_with_atom_index(mol):
    '''
    Give a mol and get it's mapped form
    '''
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx()+1)
    return mol

def refine_pd_list(pd_list, drop_c):
    new_list = []
    for dataframe in pd_list:
        for col in dataframe.columns:
            if col not in drop_c:
                dataframe[col] = dataframe[col].apply(lambda x: np.array(x))
        new_list.append(dataframe)
    return new_list

def get_pd_mean(pd_list, drop_c=[]):
    """
        Calculates the mean of DataFrames in a list after optionally dropping specified columns.

        Parameters:
        pd_list (list of pandas.DataFrame): List containing DataFrames to be averaged.
        drop_c (list of str, optional): List of column names to be dropped before calculating the mean.

        Returns:
        pandas.DataFrame: A DataFrame containing the mean values of the DataFrames in pd_list,
                          with dropped columns added back if specified.
        """
    pd_list = refine_pd_list(pd_list, drop_c)
    num = len(pd_list)
    if drop_c != []:
        drop_c_value = [pd_list[0][i] for i in drop_c]
        pd_list_new = []
        for i in pd_list:
            pd_list_new.append(i.drop(columns=drop_c))

    else:
        pd_list_new = pd_list
    col_dropped = list(pd_list_new[0].columns)
    x = []
    for i in range(0, num):
        x.append(pd_list_new[i].values)
    x = np.array(x)
    mean_value = np.mean(x, axis=0)
    mean_value = [[x.tolist()for x in i] for i in mean_value]
    new_pd = pd.DataFrame(mean_value)
    new_pd.columns = col_dropped
    if drop_c != []:
        for idx, col in enumerate(drop_c):
            new_pd.insert(0, col, drop_c_value[idx])
    return new_pd

def get_pd_variance(pd_list, drop_c=[]):

    pd_list = refine_pd_list(pd_list, drop_c)
    num = len(pd_list)
    if drop_c != []:
        drop_c_value = [pd_list[0][i] for i in drop_c]
        pd_list_new = []
        for i in pd_list:
            pd_list_new.append(i.drop(columns=drop_c))

    else:
        pd_list_new = pd_list
    col_dropped = list(pd_list_new[0].columns)
    x = []
    for i in range(0, num):
        x.append(pd_list_new[i].values)
    x = np.array(x)
    mean_value = np.var(x, axis=0)
    new_pd = pd.DataFrame(mean_value)
    new_pd.columns = col_dropped
    if drop_c != []:
        for idx, col in enumerate(drop_c):
            new_pd.insert(0, col, drop_c_value[idx])
    return new_pd

def add_target_col(pd, smi_col='Smiles'):

    atom_labels = ['atom_{}'.format(str(i)) for i in range(9)]
    bond_labels = ['bond_{}'.format(str(i)) for i in range(9)]
    smi = pd[smi_col]
    atom_num = [Chem.MolFromSmiles(i).GetNumAtoms() for i in smi]
    bond_num = [Chem.MolFromSmiles(i).GetNumBonds() for i in smi]
    atom_list = [[0] * i for i in atom_num]
    bond_list = [[0] * i for i in bond_num]
    for idx, atom_c in enumerate(atom_labels):
        pd[atom_c] = atom_list
    for idx, bond_c in enumerate(bond_labels):
        pd[bond_c] = bond_list
    return pd
def add_target_col_sub(df, smi_col='Smiles'):

    col_label = ['CYP1A2', 'CYP2A6', 'CYP2B6', 'CYP2C8', 'CYP2C9', 'CYP2C19', 'CYP2D6', 'CYP2E1', 'CYP3A4']
    smi_length = len(df[smi_col])

    df2 = pd.DataFrame()
    df2['Smiles'] = df[smi_col]
    for col in col_label:
        df2[col] = [0]*smi_length
    return df2

def prepare_data(datapath):
    example = pd.read_csv(datapath, index_col=False)
    example = add_target_col(example)
    return example

def find_index(lis, element):
    x = [i for i, l in enumerate(lis) if l == element]
    return x

def get_reaction_type(smi, target_onehot, flag):
    """
    Determine the type of chemical reaction based on the molecular structure and specified targets.

    This function analyzes a molecule's structure, provided as a SMILES string, and identifies the type of chemical
    reaction applicable based on the target atoms or bonds indicated by a one-hot encoded list. The function supports
    differentiating between several reaction types, such as rearrangement, epoxidation, oxidation, reduction, and more,
    based on the molecular configuration and the specified transformation type (atom or bond focused).

    Parameters:
    smi (str): The SMILES string representing the molecular structure.
    target_onehot (str or list): A string representation of a one-hot encoded list or list indicating the target atoms or bonds for reaction.
    flag (str): A flag indicating whether the focus is on atoms ('atom') or bonds ('bond') for determining the reaction type.

    Returns:
    list: A list of integers where each integer represents a type of reaction identified for the respective target.
          The reaction types are encoded as integers, each corresponding to a specific chemical transformation.
    """
    if type(target_onehot) == list:
        target_onehot = target_onehot
    else:
        target_onehot = literal_eval(target_onehot)
    reaction_type = []
    if sum(target_onehot) != 0:
        mol = Chem.MolFromSmiles(smi)
        if flag == 'bond':
            bond_index = find_index(target_onehot, 1)
            for index in bond_index:
                bond = mol.GetBondWithIdx(index)
                atom1 = bond.GetBeginAtom()
                atom2 = bond.GetEndAtom()
                rea_flag1 = atom1.GetSymbol() in ['N' or 'O']
                rea_flag1_ = atom2.GetSymbol() in ['N' or 'O']
                rea_flag2 = atom2.GetHybridization() == rdkit.Chem.rdchem.HybridizationType.SP2
                rea_flag2_ = atom1.GetHybridization() == rdkit.Chem.rdchem.HybridizationType.SP2
                epo_flag = bond.GetBondType() == Chem.rdchem.BondType.DOUBLE
                oxi_flag1 = atom1.GetSymbol() == 'O' and atom1.GetDegree() == 1
                oxi_flag1_ = atom2.GetSymbol() == 'O' and atom2.GetDegree() == 1
                oxi_flag2 = bond.GetBondType() == Chem.rdchem.BondType.SINGLE
                if (rea_flag1 and rea_flag2) or (rea_flag1_ and rea_flag2_):  # rearrangement
                    reaction_type.append(1)
                elif epo_flag:  # epoxidation
                    reaction_type.append(3)
                elif (oxi_flag1 or oxi_flag1_) and oxi_flag2:
                    reaction_type.append(2)
                else:
                    reaction_type.append(5)
        elif flag == 'atom':
            atom_index = find_index(target_onehot, 1)
            for index in atom_index:
                atom = mol.GetAtomWithIdx(index)
                red_flag = atom.GetSymbol() == 'N'
                red_flag2 = 'O' in [i.GetSymbol() for i in atom.GetNeighbors()]
                heto_flag = atom.GetSymbol() in ['N', 'S', 'P']
                co_flag = atom.GetSymbol() == 'C'
                if red_flag and red_flag2:
                    reaction_type.append(0)
                elif heto_flag:
                    reaction_type.append(4)
                elif co_flag:
                    reaction_type.append(6)
        else:
            reaction_type.append(7)
    else:
        reaction_type.append(8)
    return reaction_type

