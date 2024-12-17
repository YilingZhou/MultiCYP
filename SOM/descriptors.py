import subprocess
import pandas as pd
from ast import literal_eval
from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np
def fix_atom_descriptor(smi ,mapped_index ,atom_descriptors):
    # make sure atom descriptors have the same index order with the original smi
    mapped_index = literal_eval(mapped_index.replace('=', ':'))
    atom_descriptors = literal_eval(atom_descriptors)
    descriptor_list = []
    for idx, values in enumerate(atom_descriptors):
        descriptor_list.append((values, mapped_index[idx +1]))
    fix_descriptor = list([i for i in zip(*sorted(descriptor_list, key=lambda x: x[1]))][0])
    return fix_descriptor

def add_atom_descriptor(original_pd ,atommap ,smi_col='Smiles'):
    # get an original Dataframe with smi col
    # return a Dataframe append atom descriptors calculate by CDK
    EffectiveAtomPolarizability = []
    PartialSigmaCharge = []
    PartialTChargeMMFF94 = []
    PiElectronegativity = []
    SigmaElectronegativity = []
    ProtonAffinityHOSE = []
    PartialTChargePEOE = []
    for idx ,smi in enumerate(original_pd[smi_col]):
        mol = Chem.MolFromSmiles(smi)
        numatom = mol.GetNumAtoms()
        atommapper = atommap + smi
        p = subprocess.Popen(atommapper, shell=True, stdout=subprocess.PIPE)
        (stdoutput, erroutput) = p.communicate()
        try:
            stdoutput = stdoutput.decode()
            stdoutput = stdoutput.split('\r\n')[:-1]
            EffectiveAtomPolarizability.append(fix_atom_descriptor(smi, stdoutput[0], stdoutput[1]))
            PartialSigmaCharge.append(fix_atom_descriptor(smi, stdoutput[0], stdoutput[2]))
            PartialTChargeMMFF94.append(fix_atom_descriptor(smi, stdoutput[0], stdoutput[3]))
            PiElectronegativity.append(fix_atom_descriptor(smi, stdoutput[0], stdoutput[4]))
            SigmaElectronegativity.append(fix_atom_descriptor(smi, stdoutput[0], stdoutput[5]))
            ProtonAffinityHOSE.append(fix_atom_descriptor(smi, stdoutput[0], stdoutput[6]))
            PartialTChargePEOE.append(fix_atom_descriptor(smi, stdoutput[0], stdoutput[7]))
        except:
            print(idx)
            EffectiveAtomPolarizability.append([0 ] *numatom)
            PartialSigmaCharge.append([0 ] *numatom)
            PartialTChargeMMFF94.append([0 ] *numatom)
            PiElectronegativity.append([0 ] *numatom)
            SigmaElectronegativity.append([0 ] *numatom)
            ProtonAffinityHOSE.append([0 ] *numatom)
            PartialTChargePEOE.append([0 ] *numatom)
    original_pd['EffectiveAtomPolarizability'] = EffectiveAtomPolarizability
    original_pd['PartialSigmaCharge'] = PartialSigmaCharge
    original_pd['PartialTChargeMMFF94'] = PartialTChargeMMFF94
    original_pd['PiElectronegativity'] = PiElectronegativity
    original_pd['SigmaElectronegativity'] = SigmaElectronegativity
    original_pd['ProtonAffinityHOSE'] = ProtonAffinityHOSE
    original_pd['PartialTChargePEOE'] = PartialTChargePEOE
    return original_pd

def add_bond_descriptor(original_pd ,atommap ,smi_col='Smiles'):
    # get a original Dataframe with smi col
    # return a Dataframe append bond descriptors calculate by CDK
    BondPartialTChargeDescriptor = []
    BondSigmaElectronegativityDescriptor = []
    BondPartialPiChargeDescriptor = []
    for idx ,smi in enumerate(original_pd[smi_col]):
        mol = Chem.MolFromSmiles(smi)
        numbond = mol.GetNumBonds()
        atommapper = atommap + smi
        p = subprocess.Popen(atommapper, shell=True, stdout=subprocess.PIPE)
        (stdoutput, erroutput) = p.communicate()
        try:
            stdoutput = stdoutput.decode()
            stdoutput = stdoutput.split('\r\n')[:-1]
            BondPartialTChargeDescriptor.append(list(literal_eval(stdoutput[0])))
            BondSigmaElectronegativityDescriptor.append(list(literal_eval(stdoutput[1])))
            BondPartialPiChargeDescriptor.append(list(literal_eval(stdoutput[2])))
        except:
            print(idx)
            BondPartialTChargeDescriptor.append([0 ] *numbond)
            BondSigmaElectronegativityDescriptor.append([0 ] *numbond)
            BondPartialPiChargeDescriptor.append([0 ] *numbond)

    original_pd['BondPartialTChargeDescriptor'] = BondPartialTChargeDescriptor
    original_pd['BondSigmaElectronegativityDescriptor'] = BondSigmaElectronegativityDescriptor
    original_pd['BondPartialPiChargeDescriptor'] = BondPartialPiChargeDescriptor
    return original_pd

def molecule_descriptors(smi):
    # compute molecule descriptors

    mol = Chem.MolFromSmiles(smi)
    atomnum = mol.GetNumAtoms()
    BCUT = [Descriptors.BCUT2D_CHGHI(mol)] *atomnum
    TPSA = [Descriptors.TPSA(mol)] *atomnum
    log = [Descriptors.MolLogP(mol)] *atomnum
    csp3 = [Chem.Lipinski.FractionCSP3(mol)] *atomnum
    mol_vt = [Descriptors.MolWt(mol)] *atomnum
    return BCUT, TPSA, log, csp3, mol_vt

def add_mol_descriptors(a, smi_col='Smiles'):
    a['BCUT'] = a[smi_col].apply(lambda x: molecule_descriptors(x)[0])
    a['TPSA'] = a[smi_col].apply(lambda x: molecule_descriptors(x)[1])
    a['log'] = a[smi_col].apply(lambda x: molecule_descriptors(x)[2])
    a['csp3'] = a[smi_col].apply(lambda x: molecule_descriptors(x)[3])
    a['mol_vt'] = a[smi_col].apply(lambda x: molecule_descriptors(x)[4])
    return a
def to_pkl(dsp_pd, dsp_col, savepath,smi_col='Smiles'):
    # get a dataframe with atomdescriptor
    # return .npz 2D array for each molecule (atom_dim, descriptor_dim)
    descriptor_array = pd.DataFrame()
    for descriptor_col in dsp_col:
        try:
            descriptor_array[descriptor_col] = [np.array(literal_eval(i)) for i in dsp_pd[descriptor_col]]
        except:
            descriptor_array[descriptor_col] = [np.array(i) for i in dsp_pd[descriptor_col]]
    descriptor_array.index = dsp_pd[smi_col]
    descriptor_array.to_pickle(savepath)
    return descriptor_array

def normalization(test_pd, col_descriptor):
    # Normalizes the numerical values in specified columns of a pandas DataFrame to a 0-1 range.
    for col in col_descriptor:
        for idx, i in enumerate(test_pd[col]):
            try:
                literal_eval(i)
            except:
                print(idx, i)
        all_scores = [literal_eval(i)for i in test_pd[col]]
        # all_scores = np.array(all_scores)
        all_scores = sum(all_scores, [])
        # print(all_scores)
        max_ = max(all_scores)
        min_ = min(all_scores)
        test_pd[col] = test_pd[col].apply(lambda x: [to_0_1(i, max_, min_) for i in literal_eval(x)])
    return test_pd

def to_0_1(x, max_, min_):
    x = (x - min_)/(max_ - min_ + 0.0000001)
    return x

def comput_descriptors(file):
    """
    Compute molecular descriptors for a given file containing SMILES strings.

    This function processes a file or a DataFrame containing SMILES (Simplified Molecular Input Line Entry System) strings
    of molecules. It first checks if the input is a DataFrame or a file path. If it is a file path, it reads the file into
    a DataFrame. Then, it uses an external Java tool (atom_descriptor.jar) to calculate atom-level descriptors. After
    obtaining atom descriptors, it further computes molecule-level descriptors.

    Parameters:
    file (str or pd.DataFrame): A file path to a CSV containing SMILES strings or a DataFrame with SMILES strings.

    Returns:
    pd.DataFrame: A DataFrame enriched with both atom-level and molecule-level descriptors.
    """

    atommapper = 'java -jar ./Descriptor/atom_descriptor.jar '
    if type(file) == pd.core.frame.DataFrame:
        smiles_csv = file
    else:
        smiles_csv = pd.read_csv(file, index_col=False)
    atom_csv = add_atom_descriptor(smiles_csv, atommapper)
    atom_mol_csv = add_mol_descriptors(atom_csv)

    return atom_mol_csv


if __name__ == '__main__':
    test_file = './test_smiles.csv'
    output = comput_descriptors(test_file)
    output.to_csv('./testfile/test_smiles_adddescriptors.csv', index=False)


