# MultiCYP
MultiCYP is an end-to-end, integrated metabolic characterization prediction model based on multi-task strategy and multi-scale features. This model provides comprehensive predictions about metabolic enzymes, metabolic sites, and metabolites for specified molecules. The methodology is described in detail in the paper.
## Installation
``` python
conda activate -n MultiCYP python=3.8
conda activate MultiCYP
conda install rdkit
conda install pytorch=1.13.0
pip install pandas
pip install numpy
pip install chemprop
```
If use multi-scale feature, JAVA [![Java](https://img.shields.io/badge/Java-21.0.2%2B-brightgreen.svg)](https://www.oracle.com/java/technologies/downloads/#java21) is needed.
## Predicting metabolites
### Prepare input files
MultiCYP supports multiple input formats, including:  
- Single SMILES string  
- SMI file  
- CSV file (must contain a column named 'Smiles')  
- SDF file

Basic Model (Molecular Graph Only):
```bash  
python predict_main.py -i input_file -o output_path  
```
To enable multi-scale descriptors, use:  
```bash  
python predict_main.py -i input_file -o output_path -a True -b True -n True  
```
**Note**:   
- The basic command uses molecular graph features only by default  
- Use the extended command with `-a True -b True -n True` flags to enable multi-scale descriptors
