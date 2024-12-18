from ast import literal_eval
from .utils import get_reaction_type
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Geometry import Point2D


rxn_type_dix= {0: 'Reduction', 1: 'Rearrangement', 2: 'C-oxidation', 3: 'Epoxidation',
               4: 'SNP-oxidation', 5: "Cleavage", 6: 'Hydroxylation'}

def draw_metabolism(smi='',atom_target= [], bond_target= []):
    bond_type = get_reaction_type(smi, bond_target, flag='bond')
    atom_type = get_reaction_type(smi, atom_target, flag='atom')
    atom_idx = [index for (index, value) in enumerate(literal_eval(atom_target)) if value == 1]
    bond_idx = [index for (index, value) in enumerate(literal_eval(bond_target)) if value == 1]
    diclofenac = Chem.MolFromSmiles(smi)
    cp = Chem.Mol(diclofenac)
    if atom_idx !=[] :
        for idx,i in enumerate(atom_idx):
            if atom_type != 8:
                name = str(rxn_type_dix[atom_type[idx]])
                cp.GetAtomWithIdx(i).SetProp("atomNote", name)
            else:
                cp.GetAtomWithIdx(i).SetProp("atomNote", "metabolism atom")
    if bond_idx != []:
        for idx, i in enumerate(bond_idx):
            if bond_type !=8:
                name = str(rxn_type_dix[bond_type[idx]])
                cp.GetBondWithIdx(i).SetProp("bondNote", name)
            else:
                cp.GetBondWithIdx(i).SetProp("bondNote", "metabolism bond")
    d2d = rdMolDraw2D.MolDraw2DSVG(500, 500)
    d2d.drawOptions().annotationFontScale = 0.71
    d2d.drawOptions().setAnnotationColour((0, 0, 0))
    d2d.drawOptions().setHighlightColour((0.1, 0.9, 0.5))
    d2d.DrawMolecule(cp, highlightAtoms=atom_idx, highlightBonds=bond_idx)
    d2d.FinishDrawing()
    return d2d


def score_to_color(score, transparency=1):
    """
    Adjusts the red color transparency based on a score.

    Parameters:
    score (float): The score between 0 and 1.
    transparency (float): The maximum transparency level, default is 0.85.

    Returns:
    tuple: A RGBA tuple where red is always fully saturated and transparency
           is adjusted according to the score.
    """
    # Ensure transparency is within valid range
    transparency = max(0.0, min(1.0, transparency))
    # Calculate adjusted transparency based on score
    adjusted_transparency = abs(score-0.6) * 2 * transparency
    # Return RGBA color with adjusted transparency
    return (1.0, 0.0, 0.0, adjusted_transparency)


def draw_molecules_with_scores(molecule_list, file_path, img_size=(500, 450)):
    """
    Draw multiple molecules with scores and titles into a single SVG file.

    Parameters:
    molecule_list (list): List of tuples, each containing a title, SMILES string,
                          atom score dictionary, and bond score dictionary.
    img_size (tuple): The width and height of each molecule image, defaults to (500, 450).

    Returns:
    None
    """
    # Calculate the total height for the SVG canvas
    height_per_molecule = img_size[1] + 60
    if molecule_list != None: 
      total_height = height_per_molecule * len(molecule_list)
      # Initialize the global SVG canvas
      global_svg = f'<svg width="{img_size[0]}" height="{total_height}" xmlns="http://www.w3.org/2000/svg">\n'
  
      for index, (title, smiles, atom_scores, bond_scores) in enumerate(molecule_list):
          # Convert SMILES string to molecule object and compute 2D coordinates
          mol = Chem.MolFromSmiles(smiles)
          AllChem.Compute2DCoords(mol)
  
          # Create drawer object for each molecule
          drawer = rdMolDraw2D.MolDraw2DSVG(img_size[0], img_size[1])
  
          # Create color highlights based on scores for atoms and bonds
          highlight_atom_colors = {i: score_to_color(score[0], 0.8) for i, score in atom_scores.items()}
          highlight_atoms = list(highlight_atom_colors.keys())
          highlight_bond_colors = {i: score_to_color(score[0], 0.8) for i, score in bond_scores.items()}
          highlight_bonds = list(highlight_bond_colors.keys())
  
          # Modify drawing options
          options = drawer.drawOptions()
          options.addAtomIndices = False
          options.atomLabelFontSize = 16
          options.bondLineWidth = 3
          options.highlightBondWidthMultiplier = 4
          options.continuousHighlight = True
          options.setHighlightColour((1, 0, 0, 0.8))
          options.backgroundColour = (1, 1, 1)
  
          # Draw the molecule with highlights
          rdMolDraw2D.PrepareAndDrawMolecule(
              drawer, mol, highlightAtoms=highlight_atoms, highlightAtomColors=highlight_atom_colors,
              highlightBonds=highlight_bonds, highlightBondColors=highlight_bond_colors
          )
  
          # Finish drawing and retrieve SVG text
          drawer.FinishDrawing()
          svg_fragment = drawer.GetDrawingText()
  
          # Remove XML header declaration
          svg_fragment = '\n'.join(svg_fragment.split('\n')[1:])
  
          # Add labels for atoms based on their scores and types
          atom_offset = {'dx': 10, 'dy': -10}
          for i, (score, atom_type) in atom_scores.items():
              pos = drawer.GetDrawCoords(i)
              text_label = f'<text x="{pos.x + atom_offset["dx"]}" y="{pos.y + atom_offset["dy"]}" ' \
                           f'font-size="16" font-family="Arial" fill="black" text-anchor="start">' \
                           f'{score:.2f}\n{atom_type}</text>'
              svg_fragment = svg_fragment.replace('</svg>', f'{text_label}</svg>', 1)
  
              # Add labels for bonds based on their scores and types
          bond_offset = {'dx': 10, 'dy': 10}
          for i, (score, bond_type) in bond_scores.items():
              bond = mol.GetBondWithIdx(i)
              begin_atom_idx = bond.GetBeginAtomIdx()
              end_atom_idx = bond.GetEndAtomIdx()
              begin_pos = drawer.GetDrawCoords(begin_atom_idx)
              end_pos = drawer.GetDrawCoords(end_atom_idx)
              mid_pos = Point2D((begin_pos.x + end_pos.x) / 2, (begin_pos.y + end_pos.y) / 2)
              text_label = f'<text x="{mid_pos.x + bond_offset["dx"]}" y="{mid_pos.y + bond_offset["dy"]}" ' \
                           f'font-size="16" font-family="Arial" fill="black" text-anchor="start">' \
                           f'{score:.2f}\n{bond_type}</text>'
              svg_fragment = svg_fragment.replace('</svg>', f'{text_label}</svg>', 1)
  
              # Add title above each molecule image
          title_y_position = height_per_molecule * index + 30
          title_text = f'<text x="{img_size[0] / 2}" y="{title_y_position}" font-size="18" font-family="Arial" fill="black" text-anchor="middle">{title}</text>\n'
          svg_fragment = f'<g transform="translate(0, {height_per_molecule * index + 60})">\n{svg_fragment}\n</g>\n'
  
          # Append the title and molecule SVG fragment to the global SVG
          global_svg += title_text + svg_fragment
  
      global_svg += '</svg>'
  
      # Write the global SVG to a file with UTF-8 encoding
      with open(file_path, 'w', encoding='utf-8') as file:
          file.write(global_svg)
  
      #     # Display the generated SVG in the notebook
      # display(SVG(global_svg))


def draw_save(pred, var, enzymes=[],savepath=''):
    for enzyme in enzymes:
        atom_list = pred[f'atom_{enzyme}']
        bond_list = pred[f'bond_{enzyme}']
        atom_var = var
        bond_var = var
if __name__ == '__main__':
    print(1)
    # file_path = 'molecule_with_scores.svg'
    # with open(file_path, 'w') as file:
    #     file.write(svg)
    #
    #     # 显示分子图像
    # display(SVG(svg))
    molecule_list = [
        ('分子 1', 'CC1=CC(=O)NC2=CC=CC=C2C1=O',
         {
             0: [0.9, 'abc'],
             1: [0.7, 'def'],
             2: [0.75, 'ghi'],
             3: [0.65, 'jkl'],
             4: [0.55, 'mno'],
             5: [0.52, 'pqr']
         },
         {
             7: [0.55, 'vwx'],
             8: [0.58, 'yza'],
             9: [0.65, 'bcd'],
             11: [0.85, 'cde']
         }),
        ('分子 2', 'C1=CC=C(C=C1)C2=CC=CC=C2',
         {
             0: [0.5, 'xyz'],
             1: [0.6, 'uvw'],
             2: [0.7, 'rst'],
             3: [0.8, 'opq'],
             4: [0.9, 'lmn'],
             5: [1.0, 'ijk']
         },
         {
             0: [0.45, 'ghi'],
             1: [0.4, 'def'],
             2: [0.35, 'abc'],
             3: [0.3, 'xyz']
         }),
        ('分子 3', 'CC1=CC(=O)NC2=CC=CC=C2C1=O',
         {
             0: [0.9, 'abc'],
             1: [0.7, 'def'],
             2: [0.75, 'ghi'],
             3: [0.65, 'jkl'],
             4: [0.55, 'mno'],
             5: [0.52, 'pqr']
         },
         {
             7: [0.55, 'vwx'],
             8: [0.58, 'yza'],
             9: [0.65, 'bcd'],
             11: [0.85, 'cde']
         }),
    ]

    draw_molecules_with_scores(molecule_list,file_path='multiple_molecules_with_scores_and_titles2.svg', img_size=(500, 450))



