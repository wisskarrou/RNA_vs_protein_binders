import pandas as pd
import numpy as np
from faerun import Faerun
from rdkit.Chem import AllChem, rdFingerprintGenerator
from rdkit.DataStructs import BitVectToText
from matplotlib.colors import ListedColormap
import umap
from sklearn.preprocessing import StandardScaler

MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048
)

# --- Function to generate RDKit ECFP4 fingerprints ---
def get_ecfp_fingerprint(mol, radius=2, nBits=2048):
    """Generates ECFP4 (Morgan) fingerprint as a numpy array."""
    if mol is None:
        return None
    fp_array = MORGAN_GEN.GetFingerprintAsNumPy(mol)
    return fp_array

def main():
    """ The main function, now using UMAP for dimensionality reduction. """
    df = pd.read_csv("rna_protein_binders.csv").dropna(subset=["smiles"]).reset_index(drop=True)

    fps_list = []
    labels = []
    rna_binding_information = []
    smiles_for_plot = []

    total = len(df)
    print("Generating ECFP4 fingerprints...")
    for i, row in df.iterrows():
        if i % 1000 == 0 and i > 0:
            print(f"{round(100 * (i / total))}% done ...")

        # Assuming your CSV columns are consistent with the original script:
        # row[1] = mol_name, row[2] = is_rna_binder, row[3] = smiles
        mol_name = row[1]
        is_rna_binder = row[2]
        smiles = row[3]
        mol = AllChem.MolFromSmiles(smiles)

        # Apply the same basic sanity checks as the original script
        if mol and mol.GetNumAtoms() > 5 and smiles.count(".") < 2:
            fp = get_ecfp_fingerprint(mol)
            if fp is not None:
                fps_list.append(fp)
                labels.append(mol_name)
                if is_rna_binder=="Protein":
                    binding_variable = 0
                if is_rna_binder=="RNA":
                    binding_variable = 1
                if is_rna_binder=="DNA":
                    binding_variable = 2
                rna_binding_information.append(binding_variable)
                smiles_for_plot.append(smiles)
    
    # Convert list of fingerprints to a NumPy array for UMAP
    X = np.array(fps_list)
    print(f"Total molecules processed: {len(X)}")

    # ----------------------------------------------------
    # CORE REPLACEMENT: UMAP Dimensionality Reduction
    # ----------------------------------------------------

    print("Running UMAP to get 2D coordinates...")
    # Standardize the data (optional, but often helps UMAP)
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)
    
    # Initialize and fit UMAP
    # UMAP parameters are analogous to TMAP's k (n_neighbors) and layout steps
    reducer = umap.UMAP(
        n_neighbors=100,  # Similar to cfg.k in TMAP
        min_dist=0.1,
        n_components=2,
        metric='jaccard', # ECFP is a binary fingerprint, Jaccard distance is appropriate
        random_state=42
    )
    embedding = reducer.fit_transform(X)

    x = embedding[:, 0]
    y = embedding[:, 1]
    
    # TMAP generates tree structure (s, t) for plotting. UMAP does not. 
    # We set s and t to empty arrays, which tells Faerun to plot the scatter points 
    # but skip the tree (MST) component.
    s = np.array([])
    t = np.array([])

    # ----------------------------------------------------
    # Plotting Setup (remains similar to original)
    # ----------------------------------------------------

    # Create the labels and the integer encoded array for the groups
    labels_groups, groups = Faerun.create_categories(rna_binding_information)

    # Define a colormap highlighting approved vs non-approved
    # (Simplified color definition for this example)
    custom_cmap = ListedColormap(["#2ecc71", "#9b59b6"], name="custom")
    bin_cmap = ListedColormap(
        [
            '#EF3B2C',
            '#6BAED6', 
            #"#2ecc71"
        ], 
        name="bin_cmap"
    )

    f = Faerun(
        clear_color="#222222",
        coords=False,
        view="front",
        impress='UMAP Visualization'
    )

    f.add_scatter(
        "RNAvsprot",
        {
            "x": x,
            "y": y,
            "c": [groups, rna_binding_information],
            "labels": labels,
            "smiles": smiles_for_plot, # Pass smiles data for hover effect
        },
        shader="smoothCircle",
        colormap=[custom_cmap, bin_cmap],
        point_scale=0.5,  # Reduced size
        categorical=[True, True],
        has_legend=True,
        legend_labels=[
            labels_groups,
            [
                (0, "Protein binder"), 
                (1, "RNA binder"), 
                #(2, "DNA binder"),
            ],
        ],
        selected_labels=["smiles", "mol_name"],
        series_title=["Group", "Category"],
    )

    # Note: We omit f.add_tree because UMAP does not generate a tree structure.
    # We will plot the scatter plot using a custom template to enable SMILES hover.

    # We use a custom plot call to enable the smiles template even without a tree.
    # The 'rnavsprotbinderstree' name is changed to the scatter plot's name.
    f.plot("index", template="smiles")

if __name__ == "__main__":
    main()
