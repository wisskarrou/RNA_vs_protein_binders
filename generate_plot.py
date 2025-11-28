import os

import pandas as pd
import numpy as np
import scipy.stats as ss
from faerun import Faerun
from rdkit.Chem import AllChem, rdFingerprintGenerator, Lipinski, rdMolDescriptors, Descriptors
from rdkit import Chem
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors
import umap
from sklearn.manifold import TSNE


TSNE_PERPLEXITY = 50 # Typical value between 5 and 50
RANDOM_STATE = 42

def hariboss_drug_like_filter(mol):
    """Implements the drug-likeness filter defined in HARIBOSS (Panei et al., 2022): moleuclar weight between 160
    
    Returns:
        bool: True if the molecule passes HARIBOSS drug-likeness filter.
    """
    if mol is None:
        return False

    AUTHORIZED_ELEMENTS = {
        'C', 'H', 'N', 'O', 'Br', 'Cl', 'F', 'P', 'Si', 'B', 'S', 'Se'
    }

    has_carbon = False

    try:
        mass = Descriptors.ExactMolWt(mol)
    except:
        return False
    
    if mass<160 or mass>1000:
        return False
    
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()

        if symbol == 'C':
            has_carbon = True
        
        if symbol not in AUTHORIZED_ELEMENTS:
            return False 

    return has_carbon

def process_data_and_umap():
    df = pd.read_csv("rna_protein_binders.csv").dropna(subset=["smiles"]).reset_index(drop=True)
    df['binder'] = df['binder'].astype(str).str.strip() # Clean column
    
    # Generate Fingerprints
    MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = []
    valid_indices = []
    
    # Placeholder for properties
    props = {
        'labels': [], 'binder': [], 'mw': [], 'logp': [], 
        'tpsa': [], 'smiles': [], 'mol_name': [], 'aromatic_rings': [],
        'heavy_atoms': [], 'formal_charge': [],'h_acceptors': [], 'h_donors': [],
        'lipinski': [], 'has_coc': [], 'has_tz': [], 'has_sa': [],
    }
    substruct_coc = AllChem.MolFromSmiles("COC")
    substruct_sa = AllChem.MolFromSmiles("NS(=O)=O")
    substruct_tz = AllChem.MolFromSmiles("N1N=NN=C1")
    
    # Fast loop
    for i, row in df.iterrows():
        mol = AllChem.MolFromSmiles(row['smiles'])
        if row['smiles'].count(".") < 2 and hariboss_drug_like_filter(mol):
            fp = MORGAN_GEN.GetFingerprintAsNumPy(mol)
            fps.append(fp)
            valid_indices.append(i)
            is_rna_binder = row['binder']
            if is_rna_binder=="Protein":
                    binding_variable = 0
            if is_rna_binder=="RNA":
                binding_variable = 1
            if is_rna_binder=="DNA":
                binding_variable = 2
            props['labels'].append(f"{row['smiles']}__{row['mol_name']}")
            props['binder'].append(binding_variable)
            props['mw'].append(Descriptors.MolWt(mol))
            props['logp'].append(Descriptors.MolLogP(mol))
            props['tpsa'].append(Descriptors.TPSA(mol))
            props['smiles'].append(row['smiles'])
            props['mol_name'].append(row['mol_name'])
            props['aromatic_rings'].append(Descriptors.RingCount(mol))
            props['heavy_atoms'].append(mol.GetNumHeavyAtoms())
            props['formal_charge'].append(Chem.GetFormalCharge(mol))
            props['h_acceptors'].append(Lipinski.NumHAcceptors(mol))
            props['h_donors'].append(Lipinski.NumHDonors(mol))
            props['lipinski'].append(props['mw'][-1]<=500. and props['logp'][-1]<=5 and props['h_donors'][-1]<=5 and props['h_acceptors'][-1]<=10)
            props['has_coc'].append(mol.HasSubstructMatch(substruct_coc))
            props['has_sa'].append(mol.HasSubstructMatch(substruct_sa))
            props['has_tz'].append(mol.HasSubstructMatch(substruct_tz))

    # Run UMAP
    X = np.array(fps)
    tsne = TSNE(
        n_components=2,
        perplexity=TSNE_PERPLEXITY,
        init='pca',          # Use PCA initialization for stability
        random_state=RANDOM_STATE,
        n_jobs=-1,           # Use all available cores
        metric='jaccard'     # Jaccard distance is often appropriate for binary fingerprints
    )
    
    try:
        embedding = tsne.fit_transform(X)
    except ValueError as e:
        print(f"Error during t-SNE: {e}. Check if the number of samples is greater than 3 * perplexity.")
        return
    
    # Store coordinates in the props dictionary
    props['x'] = embedding[:, 0]
    props['y'] = embedding[:, 1]
        
    # Return a DataFrame for easy filtering
    return pd.DataFrame(props)

# --- 3. Main Application ---
def main():
    
    # Load Data (Cached)
    df = process_data_and_umap()
    for property in ['mw','logp', 'tpsa', 'aromatic_rings', 'heavy_atoms', 'formal_charge', 'h_acceptors', 'h_donors']:
        df[f'normalized_{property}'] = (df[property]-df[property].min())/(df[property].max()-df[property].min())
    binder_id = {
        'Protein': 0,
        'RNA': 1
    }
    rna_bin_cmap = ListedColormap(['#426e87', '#6BAED6'], name="rna_bin_cmap")
    protein_bin_cmap = ListedColormap(['#852018', '#EF3B2C'], name="protein_bin_cmap")
    bin_cmap = {
        "RNA": rna_bin_cmap,
        "Protein": protein_bin_cmap,
    }
    continuous_cmap = {
        "RNA": "Blues",
        "Protein": "Reds",
    }
    f = Faerun(
        clear_color="#ffffff", # White background looks better in Dashboards
        coords=False,
        title="t-SNE visualization of RNA and protein binders in the chemical space",
        view="front",
        impress="Plot realized with the help of Faerun module"
    )
    titles = [
        "Lipinski",
        "Ethers",
        "Sulfonamides",
        "Tetrazoles",
        "TPSA",
        "logP",
        "Mol Weight",
        "H Acceptors",
        "H Donors",
        "Ring Count",
        "Number of heavy atoms",
        "Formal charge",
    ]
    categorical=[True, True, True, True, False, False, False, False, False, False, False, False]
    
    for binder in ['RNA', 'Protein']:

        subset = df[df['binder']==binder_id[binder]].copy()
        
        # Normalize properties for coloring (0-1)
        tpsa = list(subset['normalized_tpsa'])
        print(f"tpsa_ranked={tpsa}")
        logp = list(subset['normalized_logp'])
        print(f"logp_ranked={logp}")
        mw = list(subset['normalized_mw'])
        print(f"mw_ranked={mw}")
        h_acceptors = list(subset["normalized_h_acceptors"])
        print(f"h_acceptors={h_acceptors}")
        h_donors = list(subset["normalized_h_donors"])
        print(f"h_donors={h_donors}")
        aromatic_rings = list(subset['normalized_aromatic_rings'])
        print(f"aromatic_rings={aromatic_rings}")
        heavy_atoms = list(subset['normalized_heavy_atoms'])
        print(f"heavy_atoms={heavy_atoms}")
        formal_charge = list(subset['normalized_formal_charge'])
        print(f"formal_charge={formal_charge}")
        is_lipinski = subset['lipinski'].astype(int).tolist()
        print(f"is_lipinski={is_lipinski}")
        has_coc = subset['has_coc'].astype(int).tolist()
        print(f"has_coc={has_coc}")
        has_sa = subset['has_sa'].astype(int).tolist()
        print(f"has_sa={has_sa}")
        has_tz = subset['has_tz'].astype(int).tolist()
        print(f"has_tz={has_tz}")
        
        # Define a generic colormap
        f.add_scatter(
            binder,
            {
                "x": subset['x'].values,
                "y": subset['y'].values,
                "c": [
                    is_lipinski,
                    has_coc,
                    has_sa,
                    has_tz,
                    tpsa,
                    logp,
                    mw,
                    h_acceptors,
                    h_donors,
                    aromatic_rings,
                    heavy_atoms,
                    formal_charge,
                ],
                "labels": subset['labels'].values,
            },
            shader="smoothCircle",
            colormap=4*[bin_cmap[binder]]+8*[continuous_cmap[binder]],
            point_scale=5.0,
            categorical=categorical,
            has_legend=True,
            legend_labels=[
                [(0, "No"), (1, "Yes")],
                [(0, "No"), (1, "Yes")],
                [(0, "No"), (1, "Yes")],
                [(0, "No"), (1, "Yes")],
            ],
            series_title= titles,
            max_legend_label=[
                None,
                None,
                None,
                None,
                f"{df['tpsa'].max():.0f}", 
                f"{df['logp'].max():.1f}", 
                f"{df['mw'].max():.0f}",
                f"{df['h_acceptors'].max():.0f}",
                f"{df['h_donors'].max():.0f}",
                f"{df['aromatic_rings'].max():.0f}",
                f"{df['heavy_atoms'].max():.0f}",
                f"{df['formal_charge'].max():.0f}",
            ],
            min_legend_label=[
                None,
                None,
                None,
                None,
                f"{df['tpsa'].min():.0f}", 
                f"{df['logp'].min():.1f}", 
                f"{df['mw'].min():.0f}",
                f"{df['h_acceptors'].min():.0f}",
                f"{df['h_donors'].min():.0f}",
                f"{df['aromatic_rings'].min():.0f}",
                f"{df['heavy_atoms'].min():.0f}",
                f"{df['formal_charge'].min():.0f}",
            ],
            label_index=0,
            title_index=1,
            legend_title=12*[f"{binder} binders"],
        )

    # --- D. Render to HTML and Embed in Streamlit ---
    # Faerun writes to a file. We output to a temp file, read it, and embed it.
    plot_file = "index"
    f.plot(plot_file, template="smiles")
    
if __name__ == "__main__":
    main()