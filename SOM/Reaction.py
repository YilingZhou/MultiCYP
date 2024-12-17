from rdkit import Chem
from rdkit.Chem import rdChemReactions
from ast import literal_eval
import rdkit
import copy

reaction_smarts_list_boms = {
    1: ['[#6]1=[#6]-[#7]-[#6]=[#6]-[#6]-1>>[#6]1=[#6]-[#7]=[#6]-[#6]=[#6]-1'],
    2: ['[100C:1]=[100C:2]>>[100C:1]1-[100C:2]-[O]-1', '[100c:1]:[100c:2]>>[100CH2:1]1-[100CH2:2]-[O]-1'],
    3: ['[100CH1,100CH2:1]-[100O:2]>>[100C:1]=[100O:2]'],
    4: ['[100C:1]=[100O:2]>>[100C:1]-[100O:2]'],
    6: ['[100#7,100#8,100#16:1]-[100#6H:2]>>[100#7H,100#8H,100#16H:1].[100#6H0:2]=[O]',
        '[100#7,100#8,100#16:1]-[100#6H2:2]>>[100#7H,100#8H,100#16H:1].[100#6H:2]=[O]',
        '[100#7,100#8,100#16:1]-[100#6H3:2]>>[100#7H,100#8H,100#16H:1].[100#6H2:2]=[O]'],
    60: ['[*:3]1[*:4][100N,100O,100S:1]-[100#6H2:2][*:5]1>>[100N,100O,100S:1][*:4][*:3][*:5][100#6H:2]=[O]',
         '[*:3]1[*:4][100N,100O,100S:1]-[100#6H:2][*:5]1>>[100N,100O,100S:1][*:4][*:3][*:5][100#6H0:2]=[O]',
         '[*:3]1[*:6][*:4][100N,100O,100S:1]-[100#6H2:2][*:5]1>>[100N,100O,100S:1][*:4][*:6][*:3][*:5][100#6H:2]=[O]',
         '[*:3]1[*:6][*:4][100N,100O,100S:1]-[100#6H:2][*:5]1>>[100N,100O,100S:1][*:4][*:6][*:3][*:5][100#6H0:2]=[O]',
         '[*:3]1[*:6][*:7][*:4][100N,100O,100S:1]-[100#6H2:2][*:5]1>>[100N,100O,100S:1][*:4][*:7][*:6][*:3][*:5][100#6H:2]=[O]',
         '[*:3]1[*:6][*:7][*:4][100N,100O,100S:1]-[100#6H:2][*:5]1>>[100N,100O,100S:1][*:4][*:7][*:6][*:3][*:5][100#6H0:2]=[O]',
         '[*:3]1[*:6][*:7][*:8][*:4][100N,100O,100S:1]-[100#6H2:2][*:5]1>>[100N,100O,100S:1][*:4][*:8][*:7][*:6][*:3][*:5][100#6H:2]=[O]',
         '[*:3]1[*:6][*:7][*:8][*:4][100N,100O,100S:1]-[100#6H:2][*:5]1>>[100N,100O,100S:1][*:4][*:8][*:7][*:6][*:3][*:5][100#6H0:2]=[O]'],
    61: ['[100N:1]-[100S:2]>>[100N:1].[100S:2]=[O]'],
    5: ['[100C:1](=[O])[100N,100O:2]>>[100C:1](=[O])[OH].[100N,100O:2]'],
    50: ['[*:1]1[*:2][100C:0](=[O])[100N,100O:5][*:3]1>>[OH][100C:0](=[O])[*:2][*:1][*:3][100N,100O:5]',
        '[*:1]1[*:4][*:2][100C:0](=[O])[100N,100O:6][*:3]1>>[OH][100C:0](=[O])[*:2][*:4][*:1][*:3][100N,100O:6]',
        '[*:1]1[*:4][*:5][*:2][100C:0](=[O])[100N,100O:7][*:3]1>>[OH][100C:0](=[O])[*:2][*:5][*:4][*:1][*:3][100N,100O:7]',
        '[*:1]1[*:4][*:5][*:6][*:2][100C:0](=[O])[100N,100O:8][*:3]1>>[OH][100C:0](=[O])[*:2][*:6][*:5][*:4][*:1][*:3][100N,100O:8]'],
    7: ['[100S,100P,100N:1](=[O])-[100O:2]>>[100S,100P,100N:1](=[O])[OH].[100O:2]',
        '[100S,100P,100N:1](=[S])-[100O:2]>>[100S,100P,100N:1](=[S])[OH].[100O:2]' ],
    70: ['[*:3]1[*:4][100S,100P,100N:1](=[O])-[O:2][*:5]1>>[OH][100S,100P,100N:1](=[O])[*:4][*:3][*:5][100O:2]',
         '[*:3]1[*:6][*:4][100S,100P,100N:1](=[O])-[O:2][*:5]1>>[OH][100S,100P,100N:1](=[O])[*:4][*:6][*:3][*:5][100O:2]',
         '[*:3]1[*:6][*:7][*:4][100S,100P,100N:1](=[O])-[O:2][*:5]1>>[OH][100S,100P,100N:1](=[O])[*:4][*:7][*:6][*:3][*:5][100O:2]',
         '[*:3]1[*:6][*:7][*:8][*:4][100S,100P,100N:1](=[O])-[O:2][*:5]1>>[OH][100S,100P,100N:1](=[O])[*:4][*:8][*:7][*:6][*:3][*:5][100O:2]'],
    8: ['[100N:1]=[100N:2]>>[100N:1].[100N:2]'],
    9: ['[100C,100P:1]=[100S:2]>>[100C,100P:1]=[100O:2]'],
    10: ['[100C:1]-[100Cl:2]>>[100C:1]=[100O:2]'],
    11: ['[100#6:1]-[100Cl,100Br,100F,100I:2]>>[100C:1]-[100O:2]'],
    12: ['[*:3]1[*:4][100N,100O,100S:1]-[100#6H2:2][*:5]1>>[100N,100O,100S:1][*:4][*:3][*:5][100#6H:2]=[O]',
         '[*:3]1[*:4][100N,100O,100S:1]-[100#6H:2][*:5]1>>[100N,100O,100S:1][*:4][*:3][*:5][100#6H0:2]=[O]']
}

# 'C1C[O:2][C:1][O:3]1>>[O:3]CC[O:2].O=[C:1]=O'

reaction_smarts_list_aoms = {
    1: ['[100#7+1:1](=[O])[O]>>[100#7+0:1]([H])[O][H]', '[100#7+1:1](=[O])[O-]>>[100#7+0:1]([H])[H]'],
    2: ['[100#7:1]>>[100#7+1:1][O-]'],
    3: ['[100ND1:1]>>[100N:1]([H])[OH]',
        '[100ND2:1]>>[100N:1][OH]'],
    4: ['[100S,100P:1]>>[100S,100P:1]=O'],
    5: ['[100C:1]=[O]>>[100C:1](=[O])[OH]'],
    6: ['[100#6H1,100#6H2,100#6H3:1]>>[100#6:1][OH]']
}

reaction_type_boms = {
    0: 'others',
    1: 'Rearrangement',
    2: 'Epoxidation',
    3: 'C-O Oxidation',
    4: 'C=O Reduction',
    6: 'Dealkylation',
    60: 'Dealkylation',
    61: 'Dealkylation',
    5: 'Hydrolysis1',
    50: 'Hydrolysis1',
    7: 'Hydrolysis2',
    70: 'Hydrolysis2',
    8: 'azo-Reduction',
    9: 'Desulfurization',
    10: 'Geminal Halide oxidation',
    11: 'Dehalogenation',
    12: 'Dioxane-Dealkylation'
}

reaction_type_aoms = {
    0: 'others',
    1: 'NO2-Reduction',
    2: 'N-Oxidation',
    3: 'N-Hydroxylation',
    4: 'S/P-Oxidation',
    5: 'C=O Oxidation',
    6: 'Hydroxylation'
}


def find_index(lis, element):
    """
    This function finds all indices of a given element in a list and returns them as a list.

    Parameters:
    lis (list): The list in which to search for the element.
    element (any): The element whose indices need to be found in the list.

    Returns:
    list: A list of indices where the element is found in the given list.
    """

    # List comprehension to find all indices where the element matches
    x = [i for i, l in enumerate(lis) if l == element]

    # Return the list of indices
    return x

def is_carbon_in_X_double_O(smiles, carbon_index):
    """
    This function checks if a given carbon atom, identified by its index, is part of a substructure where it is
    double-bonded to an oxygen, sulfur, phosphorus, or nitrogen atom. The function uses SMARTS pattern matching.

    Parameters:
    smiles (str): The SMILES string representing the molecule.
    carbon_index (int): The index of the carbon atom to be checked.

    Returns:
    bool: True if the carbon atom is part of the X=O substructure, False otherwise.
    """

    mol = Chem.MolFromSmiles(smiles)

    # SMARTS pattern to match C=S, C=O, P=O, and N=O substructures
    pattern = Chem.MolFromSmarts("[C,S,P,N]=[O]")

    # Get all substructure matches for the given pattern
    matches = mol.GetSubstructMatches(pattern)
    for match in matches:
        # Check if the carbon index is part of the match
        if carbon_index in match:
            return True

    return False

def is_X_in_Rearrangement(smiles, carbon_index):
    """
    This function checks if a given atom, identified by its index, is part of a cyclic substructure with the pattern
    [N,O]1C=CCC=C1. The function uses SMARTS pattern matching to identify such substructures.

    Parameters:
    smiles (str): The SMILES string representing the molecule.
    carbon_index (int): The index of the atom to be checked.

    Returns:
    bool: True if the atom is part of the specified cyclic substructure, False otherwise.
    """

    mol = Chem.MolFromSmiles(smiles)

    # SMARTS pattern to match the cyclic rearrangement structure
    pattern = Chem.MolFromSmarts("[N,O]1C=CCC=C1")

    # Get all substructure matches for the given pattern
    matches = mol.GetSubstructMatches(pattern)
    for match in matches:
        # Check if the atom index is part of the match
        if carbon_index in match:
            return True

    return False

def is_in_dioxane(smiles, carbon_index):
    """
    This function checks if a given atom, identified by its index, is part of a dioxane-like substructure, defined as
    [rO]-[rC]-[rO]. The function uses SMARTS pattern matching to identify such substructures.

    Parameters:
    smiles (str): The SMILES string representing the molecule.
    carbon_index (int): The index of the atom to be checked.

    Returns:
    bool: True if the atom is part of the dioxane substructure, False otherwise.
    """

    mol = Chem.MolFromSmiles(smiles)

    # SMARTS pattern to match the dioxane-like structure
    pattern = Chem.MolFromSmarts("[rO]-[rC]-[rO]")

    # Get all substructure matches for the given pattern
    matches = mol.GetSubstructMatches(pattern)
    for match in matches:
        # Check if the atom index is part of the match
        if carbon_index in match:
            return True

    return False

def get_reaction_type2(smi, target_onehot, flag):

    """
    This function identifies the reaction types based on the input SMILES string and the target onehot vector.

    Parameters:
    smi (str): The SMILES representation of the molecule.
    target_onehot (list or str): The onehot vector indicating the targets. If it is a string, it will be converted to a list.
    flag (str): A flag to indicate whether to look at 'bond' or 'atom'.

    Returns:
    list: A list of reaction type integers.
    """

    if type(target_onehot) == str:
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

                # Rearrangement
                rea_flag1 = atom1.GetSymbol() in ['N', 'O'] and is_X_in_Rearrangement(smi, atom1.GetIdx())
                rea_flag1_ = atom2.GetSymbol() in ['N', 'O'] and is_X_in_Rearrangement(smi, atom2.GetIdx())
                rea_flag2 = atom2.GetHybridization() == rdkit.Chem.rdchem.HybridizationType.SP2
                rea_flag2_ = atom1.GetHybridization() == rdkit.Chem.rdchem.HybridizationType.SP2
                rea_flag3 = bond.GetBondType() == Chem.rdchem.BondType.SINGLE

                # Epoxidation
                epo_flag1 = bond.GetBondType() == Chem.rdchem.BondType.DOUBLE or bond.GetBondType() == Chem.rdchem.BondType.AROMATIC
                epo_flag2 = atom1.GetSymbol() == 'C' and atom2.GetSymbol() == 'C'

                # c-o oxidation
                oxi_flag1 = atom1.GetSymbol() == 'O' and atom1.GetDegree() == 1
                oxi_flag1_ = atom2.GetSymbol() == 'O' and atom2.GetDegree() == 1
                oxi_flag2 = bond.GetBondType() == Chem.rdchem.BondType.SINGLE

                # C=O Reduction
                COReduction_flag1 = bond.GetBondType() == Chem.rdchem.BondType.DOUBLE
                COReduction_flag2 = (atom1.GetSymbol() == 'O' and atom2.GetSymbol() == 'C') or (
                            atom2.GetSymbol() == 'O' and atom1.GetSymbol() == 'C')

                # Dioxane-Dealkylation
                DD_flag1 = is_in_dioxane(smi, atom1.GetIdx()) or is_in_dioxane(smi, atom2.GetIdx())

                # Dealkylation
                clear_flag1 = atom1.GetSymbol() == 'C' and atom2.GetSymbol() in ['N', 'O', 'P', 'S']
                clear_flag2 = atom2.GetSymbol() == 'C' and atom1.GetSymbol() in ['N', 'O', 'P', 'S']
                clear_flag3 = bond.GetBondType() == Chem.rdchem.BondType.SINGLE
                clear_ring_flag = atom1.IsInRing() and atom2.IsInRing()
                clear_flagSN = (atom1.GetSymbol() == 'S' and atom2.GetSymbol() == 'N') or (
                            atom2.GetSymbol() == 'S' and atom1.GetSymbol() == 'N')

                # Hydrolysis1
                Hydro1_flag1 = bond.GetBondType() == Chem.rdchem.BondType.SINGLE
                Hydro1_flag2 = (atom1.GetSymbol() == 'C' and atom2.GetSymbol() in ['N', 'O']) or (
                            atom2.GetSymbol() == 'C' and atom1.GetSymbol() in ['N', 'O'])
                Hydro1_flag3 = is_carbon_in_X_double_O(smi, atom1.GetIdx()) or is_carbon_in_X_double_O(smi,
                                                                                                       atom2.GetIdx())
                Hydro1_ring_flag = atom1.IsInRing() and atom2.IsInRing()

                # Hydrolysis2
                Hydro2_flag1 = bond.GetBondType() == Chem.rdchem.BondType.SINGLE
                Hydro2_flag2 = (atom1.GetSymbol() == 'O' and atom2.GetSymbol() in ['S', 'P', 'N']) or (
                            atom2.GetSymbol() == 'O' and atom1.GetSymbol() in ['S', 'P', 'N'])
                Hydro2_flag3 = is_carbon_in_X_double_O(smi, atom1.GetIdx()) or is_carbon_in_X_double_O(smi,
                                                                                                       atom2.GetIdx())
                Hydro2_ring_flag = atom1.IsInRing() and atom2.IsInRing()

                # azo-Reduction
                azo_flag1 = bond.GetBondType() == Chem.rdchem.BondType.DOUBLE
                azo_flag2 = atom1.GetSymbol() == 'N' and atom2.GetSymbol() == 'N'

                # Desulfurization
                DEsulf_flag1 = bond.GetBondType() == Chem.rdchem.BondType.DOUBLE
                DEsulf_flag2 = (atom1.GetSymbol() in ['C', 'P'] and atom2.GetSymbol() == 'S') or (
                        atom2.GetSymbol() in ['C', 'P'] and atom1.GetSymbol() == 'S')

                # Geminal Halide oxidation
                GHO_flag1 = bond.GetBondType() == Chem.rdchem.BondType.SINGLE
                GHO_flag2 = (atom1.GetSymbol() == 'C' and atom2.GetSymbol() in ['Cl', 'Br']) or (
                            atom2.GetSymbol() == 'C' and atom1.GetSymbol() in ['Cl', 'Br'])
                GHO_flag3 = sum([1 if i.GetSymbol() in ['Cl', 'Br'] else 0 for i in atom1.GetNeighbors()]) == 2

                # Dehalogenation
                oxif_flag1 = atom1.GetSymbol() == 'C' and atom2.GetSymbol() in ['F', 'Cl', 'Br', 'I']
                oxif_flag2 = atom2.GetSymbol() == 'C' and atom1.GetSymbol() in ['F', 'Cl', 'Br', 'I']

                if (rea_flag1 and rea_flag2 and rea_flag3) or (
                        rea_flag1_ and rea_flag2_ and rea_flag3):  # rearrangement
                    reaction_type.append(1)
                elif epo_flag1 and epo_flag2:  # epoxidation
                    reaction_type.append(2)
                elif (oxi_flag1 or oxi_flag1_) and oxi_flag2:  # c-o oxidation
                    reaction_type.append(3)
                elif COReduction_flag1 and COReduction_flag2:  # C=O Reduction
                    reaction_type.append(4)
                elif Hydro1_flag1 and Hydro1_flag2 and Hydro1_flag3:  # Hydrolysis1
                    if Hydro1_ring_flag:
                        reaction_type.append(50)
                    else:
                        reaction_type.append(5)
                elif DD_flag1:  # Dioxane-Dealkylation
                    reaction_type.append(12)
                elif (clear_flag1 and clear_flag3) or (clear_flag2 and clear_flag3):  # Dealkylation
                    if clear_ring_flag:
                        reaction_type.append(60)
                    else:
                        reaction_type.append(6)
                elif Hydro2_flag1 and Hydro2_flag2:  # Hydrolysis2
                    if Hydro2_ring_flag:
                        reaction_type.append(70)
                    else:
                        reaction_type.append(7)
                elif azo_flag1 and azo_flag2:  # azo-Reduction
                    reaction_type.append(8)
                elif DEsulf_flag1 and DEsulf_flag2:  # Desulfurization
                    reaction_type.append(9)
                elif GHO_flag1 and GHO_flag2 and GHO_flag3:  # Geminal Halide oxidation
                    reaction_type.append(10)
                elif oxif_flag1 or oxif_flag2:  # Dehalogenation
                    reaction_type.append(11)
                elif clear_flagSN:  # Dealkylation SN
                    reaction_type.append(61)
                else:
                    reaction_type.append(0)
        elif flag == 'atom':
            atom_index = find_index(target_onehot, 1)
            for index in atom_index:
                atom = mol.GetAtomWithIdx(index)

                # NO2-Reduction
                red_flag = atom.GetSymbol() == 'N'
                red_flag2 = 'O' in [i.GetSymbol() for i in atom.GetNeighbors()]

                # N-Oxidation
                Noxi_flag1 = atom.GetSymbol() == 'N'
                Noxi_flag2 = atom.GetDegree() == 3 or atom.GetIsAromatic()

                # N-Hydroxylation
                Nhydro_flag1 = atom.GetSymbol() == 'N'
                Nhydro_flag2 = atom.GetDegree() in [2, 1]

                # S/P-Oxidation
                SPoxi_flag1 = atom.GetSymbol() in ['S', 'P']
                SPoxi_flag2 = sum([1 if i.GetSymbol() == 'O' else 0 for i in atom.GetNeighbors()]) in [1, 0]

                # C=O Oxidation
                C0oxi_flag1 = atom.GetSymbol() == 'C'
                C0oxi_flag2 = is_carbon_in_X_double_O(smi, atom.GetIdx())

                # Hydroxylation
                Hydroxy_flag1 = atom.GetSymbol() == 'C'

                if red_flag and red_flag2:  # NO2-Reduction
                    reaction_type.append(1)
                elif Noxi_flag1 and Noxi_flag2:  # N-Oxidation
                    reaction_type.append(2)
                elif Nhydro_flag1 and Nhydro_flag2:  # N-Hydroxylation
                    reaction_type.append(3)
                elif SPoxi_flag1 and SPoxi_flag2:  # S/P-Oxidation
                    reaction_type.append(4)
                elif C0oxi_flag1 and C0oxi_flag2:  # C=O Oxidation
                    reaction_type.append(5)
                elif Hydroxy_flag1:  # Hydroxylation
                    reaction_type.append(6)
                else:
                    reaction_type.append(0)
    return reaction_type


def mol_with_atom_index(mol):
    """
    This function assigns each atom in the given RDKit molecule an atom map number based on its index.
    The atom map number is set to the 1-based index of the atom (i.e., the atom's index + 1).

    Parameters:
    mol (rdkit.Chem.rdchem.Mol): The RDKit molecule whose atoms' map numbers need to be assigned.

    Returns:
    rdkit.Chem.rdchem.Mol: The RDKit molecule with updated atom map numbers based on atom indices.
    """

    # Iterate over each atom in the molecule
    for atom in mol.GetAtoms():
        # Set the atom map number to the atom's index + 1 (1-based index)
        atom.SetAtomMapNum(atom.GetIdx() + 1)

    # Return the modified molecule
    return mol


def clear_atom_map(mol):
    """
    This function clears the atom map numbers from all atoms in the given RDKit molecule.
    Atom map numbers are stored as a property called 'molAtomMapNumber' on each atom.

    Parameters:
    mol (rdkit.Chem.rdchem.Mol): The RDKit molecule whose atoms' map numbers need to be cleared.

    Returns:
    rdkit.Chem.rdchem.Mol: The RDKit molecule with all atom map numbers cleared.
    """

    # Iterate over each atom in the molecule and clear the atom map number property
    [a.ClearProp('molAtomMapNumber') for a in mol.GetAtoms()]

    # Return the modified molecule
    return mol


def reset_map(smi, som_index, flag):
    """
    This function resets and sets specific atom map numbers and isotope labels in a given molecule
    based on the reaction type, site of metabolism (SOM) index, and the specified flag.

    Parameters:
    smi (str): The SMILES representation of the molecule.
    rxn_type (int): The reaction type (not directly used in this function).
    som_index (int): The index of the site of metabolism (SOM).
    flag (str): Indicates whether the reaction is related to 'atom' or 'bond'.

    Returns:
    rdkit.Chem.rdchem.Mol: The modified RDKit molecule with updated atom map numbers and isotope labels.
    """

    # Convert SMILES string to RDKit molecule
    mol = Chem.MolFromSmiles(smi)

    # Clear existing atom map numbers from the molecule
    clear_atom_map(mol)

    if flag == 'atom':
        # For atom-related reactions, set the specified atom's map number and isotope label
        mol.GetAtomWithIdx(som_index).SetAtomMapNum(1)
        mol.GetAtomWithIdx(som_index).SetIsotope(100)

    elif flag == 'bond':
        # For bond-related reactions, find the bond with the given index
        bond = mol.GetBondWithIdx(som_index)

        # Set atom map numbers and isotope labels for both atoms involved in the bond
        mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).SetAtomMapNum(1)
        mol.GetAtomWithIdx(bond.GetEndAtomIdx()).SetAtomMapNum(1)
        mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).SetIsotope(100)
        mol.GetAtomWithIdx(bond.GetEndAtomIdx()).SetIsotope(100)

    # Return the modified molecule
    return mol


def reset_map2(smi, som_index, flag, mark=100, clears=True):

    if type(smi) == str:
        mol = Chem.MolFromSmiles(smi)
    else:
        mol = smi
    if clears:
        clear_atom_map(mol)
    if flag == 'atom':
        mol.GetAtomWithIdx(som_index).SetAtomMapNum(1)
        mol.GetAtomWithIdx(som_index).SetIsotope(mark)
    elif flag == 'bond':
        bond = mol.GetBondWithIdx(som_index)
        mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).SetAtomMapNum(1)
        mol.GetAtomWithIdx(bond.GetEndAtomIdx()).SetAtomMapNum(1)

        mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).SetIsotope(mark)
        mol.GetAtomWithIdx(bond.GetEndAtomIdx()).SetIsotope(mark)

    return mol

def remove_isotope_labels(mol):
    """
    This function removes isotope labels from all atoms in a given RDKit molecule.
    Isotope labels are often used in cheminformatics to distinguish between atoms of the same element that have different mass numbers.

    Parameters:
    mol (rdkit.Chem.rdchem.Mol): The RDKit molecule from which the isotope labels should be removed.

    Returns:
    rdkit.Chem.rdchem.Mol: The RDKit molecule with all isotope labels removed.
    """

    # Iterate over each atom in the molecule
    for atom in mol.GetAtoms():
        # Set the isotope label to 0 (remove isotope labeling)
        atom.SetIsotope(0)

    # Return the modified molecule
    return mol


def check_mapped(product):
    """
    This function checks if a product molecule has any atoms with non-zero atom map numbers.
    Atom map numbers are used to track how atoms in the reactants correspond to atoms in the products after a reaction.

    Parameters:
    product (rdkit.Chem.rdchem.Mol): The RDKit molecule to be checked.

    Returns:
    bool: Returns True if all atom map numbers are zero, indicating that no atoms are mapped.
          Returns False if any atom has a non-zero atom map number, indicating that some atoms are mapped.
    """

    # Iterate over each atom in the product molecule
    for atom in product.GetAtoms():
        # Check if the atom has a non-zero map number
        if atom.GetAtomMapNum() != 0:
            # If any atom has a non-zero map number, the product is considered mapped
            return False

    # If all atoms have zero map numbers, return True
    return True

def get_products_one(smi, rxn_type, som_index, flag):
    """
    This function generates a list of product SMILES by running the specified reaction type on the given molecule.

    Parameters:
    smi (str): The SMILES representation of the molecule.
    rxn_type (int): The reaction type to be applied.
    som_index (int): The index of the site of metabolism (SOM).
    flag (str): Indicates whether the reaction is related to 'atom' or 'bond'.

    Returns:
    list: A list of unique product SMILES generated from the reaction.
    """

    # Determine the appropriate reaction SMARTS based on the flag
    if flag == 'bond':
        reaction_smarts = reaction_smarts_list_boms[rxn_type]
    else:
        reaction_smarts = reaction_smarts_list_aoms[rxn_type]

    product_list = []

    # Iterate over each reaction SMARTS pattern
    for i in range(len(reaction_smarts)):
        reaction_smart = reaction_smarts[i]
        reaction = rdChemReactions.ReactionFromSmarts(reaction_smart)

        # Reset the mapping of SMILES and create a reactant molecule
        reactant_mol = reset_map(smi, som_index, flag)
        reactants = (reactant_mol,)

        # Run the reaction to get products
        products = reaction.RunReactants(reactants)

        # Process each product and add unique ones to the product list
        for product_set in products:
            for product in product_set:
                product = remove_isotope_labels(product)
                if check_mapped(product):
                    product_smi = Chem.MolToSmiles(product)
                    product_list.append(product_smi)

    # Remove duplicates from the product list
    product_list = list(set(product_list))

    return product_list


def get_products_flag(smi, target_onehot, flag):
    """
    This function generates a dictionary of products based on the specified onehot vector and the flag determining the type of reaction.

    Parameters:
    smi (str): The SMILES representation of the molecule.
    target_onehot (list or str): The onehot vector indicating the target sites. If it is a string, it will be converted to a list using literal_eval.
    flag (str): Indicates whether the onehot vector is related to 'atom' or 'bond'.

    Returns:
    dict: A dictionary of products where the keys are tuples of (index, reaction_type) and the values are lists of products.
    """

    # Initialize product dictionary
    product_dict = {}

    # Convert SMILES to RDKit Mol object
    mol = Chem.MolFromSmiles(smi)

    # Convert target onehot vector from string to list if needed
    if type(target_onehot) == str:
        target_onehot = literal_eval(target_onehot)

    # Get reaction types and their indices based on the flag ('atom' or 'bond')
    rxn_types = get_reaction_type2(smi, target_onehot, flag)
    som_indexs = find_index(target_onehot, 1)

    # Check if indices match the lengths of corresponding reaction types
    if len(som_indexs) != len(rxn_types):
        print(f'som_index is not match with rxn_type!')

    # Generate product dictionary
    for rxn_num in range(len(som_indexs)):
        rxn_type = rxn_types[rxn_num]
        som_index = som_indexs[rxn_num]
        product_list = get_products_one(smi, rxn_type, som_index, flag)
        if flag == 'bond':
            product_dict[(som_index, reaction_type_boms[rxn_type])] = product_list
        else:
            product_dict[(som_index, reaction_type_aoms[rxn_type])] = product_list

    return product_dict


def get_products_all(smi, onehot_atom, onehot_bond):
    """
    This function generates a dictionary of products based on the specified onehot vectors for atoms and bonds in the molecule.

    Parameters:
    smi (str): The SMILES representation of the molecule.
    onehot_atom (list or str): The onehot vector for atoms. If it is a string, it will be converted to a list using literal_eval.
    onehot_bond (list or str): The onehot vector for bonds. If it is a string, it will be converted to a list using literal_eval.

    Returns:
    dict: A dictionary of products where the keys are tuples of (index, reaction_type) and the values are lists of products.
    """

    # Convert SMILES to RDKit Mol object
    mol = Chem.MolFromSmiles(smi)

    # Convert onehot vectors from strings to lists if needed
    if type(onehot_atom) == str:
        onehot_atom = literal_eval(onehot_atom)
    if type(onehot_bond) == str:
        onehot_bond = literal_eval(onehot_bond)

    # Get reaction types for bonds and their indices
    rxn_bond_types = get_reaction_type2(smi, onehot_bond, 'bond')
    boms_indexs = find_index(onehot_bond, 1)

    # Get reaction types for atoms and their indices
    rxn_atom_types = get_reaction_type2(smi, onehot_atom, 'atom')
    aoms_indexs = find_index(onehot_atom, 1)

    # Check if indices match the lengths of corresponding reaction types
    if len(boms_indexs) != len(rxn_bond_types) or len(aoms_indexs) != len(rxn_atom_types):
        print(f'som_index is not match with rxn_type!')

    # Generate product dictionary for bonds
    product_dict_bond = {}
    for rxn_num in range(len(boms_indexs)):
        rxn_type = rxn_bond_types[rxn_num]
        som_index = boms_indexs[rxn_num]
        product_list = None
        try:
            product_list = get_products_one(smi, rxn_type, som_index, 'bond')
        except:
            # print('trouble')
            pass
        product_dict_bond[(som_index, reaction_type_boms[rxn_type])] = product_list

    # Generate product dictionary for atoms
    product_dict_atom = {}
    for rxn_num in range(len(aoms_indexs)):
        rxn_type = rxn_atom_types[rxn_num]
        som_index = aoms_indexs[rxn_num]
        product_list = None
        try:
            product_list = get_products_one(smi, rxn_type, som_index, 'atom')
        except:
            print('trouble')
        product_dict_atom[(som_index, reaction_type_aoms[rxn_type])] = product_list

    # Combine bond and atom product dictionaries
    product_dict = {**product_dict_atom, **product_dict_bond}

    return product_dict


def get_products_recursive(smi, onehot_atom, onehot_bond):

    mol = Chem.MolFromSmiles(smi)

    if type(onehot_atom) == str:
        onehot_atom = literal_eval(onehot_atom)
    if type(onehot_bond) == str:
        onehot_bond = literal_eval(onehot_bond)

    rxn_bond_types = get_reaction_type2(smi, onehot_bond, 'bond')
    boms_indexs = find_index(onehot_bond, 1)

    rxn_atom_types = get_reaction_type2(smi, onehot_atom, 'atom')
    aoms_indexs = find_index(onehot_atom, 1)

    if len(boms_indexs) != len(rxn_bond_types) or len(aoms_indexs) != len(rxn_atom_types):
        print(f'som_index is not match with rxn_type!')

    # marked substrates and rxns
    aom_reaction_smarts = []
    bom_reaction_smarts = []
    clear_atom_map(mol)
    for i, index in enumerate(aoms_indexs):
        mark = 100 * (i + 1)
        mol = reset_map2(mol, aoms_indexs[i], 'atom', mark=mark, clears=False)
        aom_reaction_smarts.extend(
            [rxn.replace('100', f'{mark}') for rxn in reaction_smarts_list_aoms[rxn_atom_types[i]]])

    for i, index in enumerate(boms_indexs):
        mark = 100 * (i + 1 + len(aoms_indexs))
        mol = reset_map2(mol, boms_indexs[i], 'bond', mark=mark, clears=False)
        bom_reaction_smarts.extend(
            [rxn.replace('100', f'{mark}') for rxn in reaction_smarts_list_boms[rxn_bond_types[i]]])

    # aom reation
    reaction_mols = [mol]
    for rxn in aom_reaction_smarts:
        reaction = rdChemReactions.ReactionFromSmarts(rxn)
        copy_reaction_mols = copy.deepcopy(reaction_mols)
        for reaction_mol in copy_reaction_mols:
            reactants = (reaction_mol,)
            products = reaction.RunReactants(reactants)
            products = list(products[0])
            reaction_mols.extend(products)
            reaction_mols = list(set(reaction_mols))
    # bom reaction
    reaction_mols = [mol]
    for rxn in bom_reaction_smarts:
        reaction = rdChemReactions.ReactionFromSmarts(rxn)
        copy_reaction_mols = copy.deepcopy(reaction_mols)
        for reaction_mol in copy_reaction_mols:
            reactants = (reaction_mol,)
            products = reaction.RunReactants(reactants)
            if products == ():
                products = []
            else:
                products = list(products[0])
            reaction_mols.extend(products)
            reaction_mols = list(set(reaction_mols))

    # dupublic
    product_mols = [remove_isotope_labels(pro) for pro in reaction_mols]
    product_mols = [clear_atom_map(pro) for pro in product_mols]
    product_smis = list(set([Chem.MolToSmiles(pro) for pro in product_mols]))
    return product_smis

if __name__ == '__main__':
    # #  Example
    # smi = 'CNCCC(OC1=CC=CC=C1C)C2=CC=CC=C2'
    # onehot_atom = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0]
    # onehot_bond = [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # a = get_products_all(smi, onehot_atom, onehot_bond)
    # print(a)
    # find bug
    smi = '[CH:1]#[C:2][c:3]1[cH:4][cH:5][cH:6][c:7]([cH:8]1)[NH:9][c:10]2[n:11][cH:12][n:13][c:14]3[cH:15][c:16]([O:17][CH2:18][CH2:19][O:20][CH3:21])[c:22]([O:23][CH2:24][CH2:25][O:26][CH3:27])[cH:28][c:29]23'
    atom_ = '[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]'
    bond_ = '[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0]'
    a = get_products_all(smi, atom_, bond_)
    print(a)



