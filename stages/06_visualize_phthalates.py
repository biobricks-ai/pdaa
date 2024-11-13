import pandas as pd
import numpy as np
import seaborn as sns
import pathlib
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Draw
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

import matplotlib.pyplot as plt
tqdm.pandas()

outdir = pathlib.Path('cache/visualize_phthalates')
tmpdir = outdir / 'temp'
tmpdir.mkdir(parents=True, exist_ok=True)

# Load the data
df = pd.read_parquet('cache/zinc_phthalates/zinc_phthalates.parquet')

# Generate fingerprints
generator = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
df['fingerprint'] = df['smiles'].progress_apply(lambda x: generator.GetFingerprint(Chem.MolFromSmiles(x)))

# Calculate similarity matrix
def calculate_similarity(fp_list):
    n = len(fp_list)
    similarity_matrix = np.zeros((n, n))
    for i in tqdm(range(n)):
        similarities = DataStructs.BulkTanimotoSimilarity(fp_list[i], fp_list[i:])
        similarity_matrix[i, i:] = similarities
        similarity_matrix[i:, i] = similarities
    return similarity_matrix

fingerprints = df['fingerprint'].tolist()
similarity_matrix = calculate_similarity(fingerprints)

# get shape of matrix
print(similarity_matrix.shape)

# Perform clustering
distance_matrix = 1 - similarity_matrix
clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.5, metric='precomputed', linkage='average')
df['cluster'] = clustering.fit_predict(distance_matrix)

# Convert fingerprints to lists of integers
df['list_fingerprint'] = df['fingerprint'].apply(lambda x: list(x.ToBitString()))
savedf = df.drop(columns=['fingerprint'])
savedf.to_parquet(tmpdir / 'phthalates_clustered.parquet')

# Plot heatmap
sample_indices = np.random.choice(distance_matrix.shape[0], size=1000, replace=False)
sample_distance_matrix = distance_matrix[np.ix_(sample_indices, sample_indices)]
sns.clustermap(sample_distance_matrix, cmap='viridis', figsize=(10, 10))
plt.title('Heatmap of Phthalates Clusters')
plt.savefig(outdir / 'phthalates_clusters_heatmap.png')

# Create a barchart of the cluster sizes
df['cluster'].nunique()

# take the 10 largest clusters and sample 5 from each
cluster_sizes = df['cluster'].value_counts().sort_values(ascending=False)
top_clusters = cluster_sizes.head(10)

sampled_molecules = []
for cluster in top_clusters.index:
    cluster_df = df[df['cluster'] == cluster]
    sampled_mols = cluster_df.sample(n=5, random_state=42)['smiles'].tolist()
    sampled_molecules.append(sampled_mols)

# Draw the sampled molecules in a grid
from rdkit.Chem import AllChem

fig, axes = plt.subplots(nrows=len(sampled_molecules), ncols=5, figsize=(15, 3 * len(sampled_molecules)))

highlight_substructure = Chem.MolFromSmiles('OC(=O)C1=CC=CC=C1C(=O)O')  # Phthalic acid

# Precompute 2D coordinates for the highlight substructure
AllChem.Compute2DCoords(highlight_substructure)

for i, mols in enumerate(sampled_molecules):
    for j, smi in enumerate(mols):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            # Align the molecule so that the highlight substructure is in a fixed position
            AllChem.Compute2DCoords(mol)
            try:
                # Aligns mol to highlight_substructure
                AllChem.GenerateDepictionMatching2DStructure(mol, highlight_substructure)
            except ValueError:
                pass  # If alignment fails, continue with the default orientation

            # Find substructure matches
            hit_atoms = mol.GetSubstructMatch(highlight_substructure)
            hit_bonds = []
            if hit_atoms:
                for bond in highlight_substructure.GetBonds():
                    start_atom = hit_atoms[bond.GetBeginAtomIdx()]
                    end_atom = hit_atoms[bond.GetEndAtomIdx()]
                    hit_bonds.append(mol.GetBondBetweenAtoms(start_atom, end_atom).GetIdx())
                # Draw molecule with a transparent background and highlight substructure
                img = Draw.MolToImage(mol, size=(200, 200), highlightAtoms=hit_atoms, highlightBonds=hit_bonds, highlightColor=(1, 0, 0), bgcolor=(0, 0, 0, 0))
            else:
                img = Draw.MolToImage(mol, size=(200, 200), bgcolor=(0, 0, 0, 0))
            axes[i, j].imshow(img)
            axes[i, j].axis('off')

# Overlay semi-transparent rectangles for row background colors
background_colors = sns.color_palette("pastel", len(sampled_molecules))
for i, color in enumerate(background_colors):
    fig.patches.extend([
        plt.Rectangle(
            (0, 1 - (i + 1) / len(sampled_molecules)),  # x, y coordinates
            1,  # Width spans the whole row
            1 / len(sampled_molecules),  # Height per row
            transform=fig.transFigure,
            color=color,
            alpha=0.3,
            zorder=0
        )
    ])

plt.tight_layout()
plt.savefig(outdir / 'sampled_molecules_grid.png')

# plot the resul