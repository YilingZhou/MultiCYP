from chemprop.utils import makedirs
from SOM.Reaction import get_reaction_type2, reaction_type_boms, reaction_type_aoms, get_products_all
from SOM import predit as som_predict
from SOM.utils import add_target_col, add_target_col_sub, get_reaction_type
from SOM.draw import draw_molecules_with_scores
from Substrates import predict as sub_predict
from Substrates import load_model as sub_model
from SOM.descriptors import comput_descriptors

import pandas as pd
import argparse
from rdkit import Chem

def substrate_result(smi_path, save_path):

    # load data
    original_data_path = save_path + "/original_sub_data.csv"
    smi_df = pd.read_csv(smi_path, index_col=False)
    smi_df = add_target_col_sub(smi_df)
    makedirs(save_path)
    smi_df.to_csv(original_data_path, index=False)

    # pred
    pred_pd, vars_pd = sub_predict.ensemble_pred(modelpath_list=sub_model.multi_model, datapath=original_data_path, varout=True)
    pred_label = sub_predict.get_pred_label(pred_pd)

    # # if save
    # pred_label.to_csv(save_path+'/pred_sub.csv', index=True)

    return pred_label


def general_enzyme_result(smi_path, save_path, atom_descriptors=False, mol_descriptors=False, normalization=False, ensemble=True):

    # load data: SMILES file\csv(Smiles)\ single SMILES\sdf
    original_data_path = save_path+"/original_som_data.csv"

    if smi_path[-3:] == 'csv':
        smi_df = pd.read_csv(smi_path, index_col=False)
    elif smi_path[-3:] == 'sdf':
        sdf_mol = Chem.SDMolSupplier(smi_path)
        sdf_smi = [Chem.MolToSmiles(i)for i in sdf_mol]
        smi_df = pd.DataFrame()
        smi_df['Smiles'] = sdf_smi
    elif smi_path[-3:] == 'smi':
        smiles_list = []
        with open(smi_path, 'r') as f:
            for line in f:
                # remove blank
                smile = line.strip()
                if smile:  # not blank
                    smiles_list.append(smile)
        smi_df = pd.DataFrame()
        smi_df['Smiles'] = smiles_list
    elif smi_path[-4] != '.':
        smi_df = pd.DataFrame()
        smi_df['Smiles'] = [smi_path]
    else:
        print('Wrong format, please check your file!')

    if atom_descriptors or mol_descriptors:
        smi_df = add_target_col(smi_df)
        smi_df = comput_descriptors(smi_df)
    else:
        smi_df = add_target_col(smi_df)
    makedirs(save_path)
    smi_df.to_csv(original_data_path, index=False)

    # set threshold
    threshold = som_predict.threshold

    # get prediction
    pred, pred_label, pred_var = som_predict.predict_csv(datapath=original_data_path, savepathpkl=save_path,
                                                   atom_descriptors=atom_descriptors, mol_descriptors=mol_descriptors,
                                                   normalization=normalization, ensemble=ensemble, threshold=threshold)

    return pred, pred_label, pred_var

def get_standard_input_draw(som_pred, som_label, som_uncertainty, sub_pred, special=True, smi_col='Smiles', substrate=[]):

    Enzymes = ['CYP1A2', 'CYP2A6', 'CYP2B6', 'CYP2C8', 'CYP2C9', 'CYP2C19', 'CYP2D6', 'CYP2E1', 'CYP3A4']
    assert som_pred[smi_col].to_list() == sub_pred.index.to_list(), 'som_pred smi and sub_pred smi must be same'
    smis = som_label[smi_col]

    molecules_input = []
    csv_output_all = []
    csv_output_individual = []

    # get pred substrate enzymes or not
    for index, smi in enumerate(smis):
        if special:
            sub_enzymes_list = sub_pred.loc[smi].tolist()
            sub_enzymes_list_index = [index for index, value in enumerate(sub_enzymes_list) if value == 1]
        else:
            sub_enzymes_list_index = [i for i in range(9)]
        if substrate != []:
            sub_enzymes_list_index = substrate

        molecule_input = []
        csv_output_individual1 = []
        for i in sub_enzymes_list_index:
            # if this enzyme is none , not include
            if sum(som_label[f'atom_{i}'][index]) == 0 and sum(som_label[f'bond_{i}'][index]) == 0:
                continue

            aom_label = som_label[f'atom_{i}'][index]  # [0,1,0,0,0,0]
            aom_idx = [index for index, value in enumerate(aom_label) if value == 1]  # [1]
            aom_type = [reaction_type_aoms[type1]for type1 in get_reaction_type2(smi, aom_label, flag='atom')]  # ['type1']
            aom_uncertainty = ['H'if som_uncertainty[f'atom_{i}'][index][x] == 1 else 'L' for x in aom_idx]  # ['uncertainty'](L,H)
            aom_scores = [som_pred[f'atom_{i}'][index][x] for x in aom_idx]  # [score]
            aom_scores = [round(num, 2) for num in aom_scores]

            bom_label = som_label[f'bond_{i}'][index]  # [0,1,0,0,0,0]
            bom_idx = [index for index, value in enumerate(bom_label) if value == 1]  # [1]
            bom_type = [reaction_type_boms[type2]for type2 in get_reaction_type2(smi, bom_label, flag='bond')]  # ['type1']
            bom_uncertainty = ['H' if som_uncertainty[f'bond_{i}'][index][x] == 1 else 'L' for x in bom_idx]  # ['uncertainty'](L,H)
            bom_scores = [som_pred[f'bond_{i}'][index][x] for x in bom_idx]  # [score]
            bom_scores = [round(num, 2) for num in bom_scores]

            aom_input = {idx: [score, rxn_type+f'({uncertainty})']for idx, score, rxn_type, uncertainty in
                         zip(aom_idx, aom_scores, aom_type, aom_uncertainty)}
            bom_input = {idx: [score, rxn_type + f'({uncertainty})'] for idx, score, rxn_type, uncertainty in
                         zip(bom_idx, bom_scores, bom_type, bom_uncertainty)}

            molecule_input.append((Enzymes[i], smi, aom_input, bom_input))


            #  cols：smiles_index，smiles，enzyme，sites，scores ，rxntype，uncertainty，prosmilist；一个site一行
            product_all = get_products_all(smi, aom_label, bom_label)
            for num in range(len(aom_idx)):
                product = product_all[(aom_idx[num], aom_type[num])]
                csv_output_all.append([index, smi, Enzymes[i], aom_idx[num], aom_scores[num], aom_type[num], aom_uncertainty[num], product])
                csv_output_individual1.append([index, smi, Enzymes[i], aom_idx[num], aom_scores[num], aom_type[num], aom_uncertainty[num], product])
            for num in range(len(bom_idx)):
                product = product_all[(bom_idx[num], bom_type[num])]
                csv_output_all.append([index, smi, Enzymes[i], bom_idx[num], bom_scores[num], bom_type[num], bom_uncertainty[num], product])
                csv_output_individual1.append([index, smi, Enzymes[i], bom_idx[num], bom_scores[num], bom_type[num], bom_uncertainty[num], product])

        if molecule_input == []:
            molecules_input.append(None)
        else:
            molecules_input.append(molecule_input)

        if csv_output_individual1 == []:
            csv_output_individual.append(None)
        else:
            csv_output_individual.append(csv_output_individual1)
    return molecules_input, csv_output_all, csv_output_individual

def predict_main(data_path, save_path, special, atom_descriptors=False, mol_descriptors=False, normalization=False, ensemble=True,substrate=[]):

    makedirs(save_path)
    original_data_path = save_path + '/' + 'original_data'

    som_pred, som_label, som_var = general_enzyme_result(smi_path=data_path, save_path=original_data_path,
                                                         atom_descriptors=atom_descriptors, mol_descriptors=mol_descriptors,
                                                         normalization=normalization, ensemble=ensemble)
    sub_pred = substrate_result(smi_path=data_path, save_path=original_data_path)

    draw_input, csv_all, csv_individual = get_standard_input_draw(som_pred, som_label, som_var, sub_pred,
                                                                  special=special, smi_col='Smiles', substrate=substrate)
    length = len(draw_input)

    for i in range(length):

        molecule_path = save_path + f'/molecule_{i}'
        makedirs(molecule_path)

        # draw svg
        try:
            draw_molecules_with_scores(draw_input[i], molecule_path+'/Product.svg')
        except:
            pass

        # csv
        #  cols：smiles_index，smiles，enzyme，sites，scores ，rxntype，uncertainty，prosmilist；一个site一行
        cols = ['SMI_Idx', 'Smiles', 'Enzyme', 'SOMs', 'Scores', 'RXN_Type', 'Uncertainty', 'Products']
        try:
            df_individual = pd.DataFrame(csv_individual[i])
            df_individual.columns = cols
            df_individual.to_csv(molecule_path+'/Products.csv', index=False)
        except:
            pass

    # csv all
    df_all = pd.DataFrame(csv_all)
    cols = ['SMI_Idx', 'Smiles', 'Enzyme', 'SOMs', 'Scores', 'RXN_Type', 'Uncertainty', 'Products']
    df_all.columns = cols
    df_all.to_csv(save_path+'/Product_all.csv', index=False)

def main():

    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('--input', '-i', required=True, help='the path of input csv')
    parser.add_argument('--output', '-o', required=True, help='the path of output')
    parser.add_argument('--special', '-s', default=False, help='whether give substrate predict')
    parser.add_argument('--atom_descriptors',  '-a', default=False, help='whether to use atom descriptors')
    parser.add_argument('--mol_descriptors', '-b', default=False, help='whether to use mol descriptors')
    parser.add_argument('--normalization', '-n', default=False, help='whether normalize mol descriptors')
    parser.add_argument('--ensemble', '-e', default=True, help='whether to use ensemble prediction')

    args = parser.parse_args()
    predict_main(data_path=args.input, save_path=args.output, special=args.special,
                 atom_descriptors=args.atom_descriptors, mol_descriptors=args.mol_descriptors,
                 normalization=args.normalization, ensemble=args.ensemble)

if __name__ == '__main__':

    # print(1)
    # predict_main(data_path='./External_example/dgm_use.csv', save_path='./External_example/dgm3',
    #                  special=False, atom_descriptors=False, mol_descriptors=False, normalization=False, ensemble=True)
    main()
