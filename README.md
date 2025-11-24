# RNA_vs_protein_binders

## About the repository
This repository accompanies the Review Machine learning for RNA-targeting drug design. It provides tools to compare the chemical properties of RNA-binding and protein-binding drug-like small molecules.
The plots have been realized using Faerun module (https://github.com/reymond-group/faerun-python.git). The protein-binding small molecules considered are the ones from PDBbind database and the RNA-binding small molecule include both the ones from PDBbind and from R-SIM database (Krishnan *et al.*, 2023).
A drug-likeness filter has been applied. As in HARIBOSS (Panei *et al.*, 2022), we retain molecules which weight is between 160 and 1000 Da and containing at least one C atom and no other atom types than C, H, N, O, Br, Cl, F, P, Si, B, S, Se.

## Details about the files
* index.html: the HTML page displaying the RNA- and protein-binding small molecules with a UMAP, along with several properties including molecular weight, number of cycles, number of hydrigen bond donors and acceptors, Lipinski rule...
* RNAvsprot_enriched.js: the JavaScript file accompanying index.html
* rna_protein_binders.csv: a curated dataset containing the list of the DNA, RNA and protein binders from PDBbind and R-SIM along with their names, SMILES representations and target category (DNA, RNA or protein)
* generate_plot.py: the script used to generate index.html and RNAvsprot_enriched.js
