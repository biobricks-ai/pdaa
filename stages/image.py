import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import py3Dmol
import io

from PIL import Image
import glob
import os

# df = pd.read_parquet('/mnt/ssd_raid/workspace-jjaramillo/PDAA/raw/zinc_phthalates').head(20)
# smis = list(df['smiles'])

# print(smis)


# filt = []

# for smi in smis:
#     mol = Chem.MolFromSmiles(smi)
#     if mol.GetNumHeavyAtoms() > 15:
#         filt.append(smi)
#     if len(filt) >= 20:
#         break
   
# img_size = (1920, 1920)

# print(filt)

# def smiles_to_3d_image(smiles, img_name):
#     print(img_name)
#     # Convert SMILES to a molecule
#     mol = Chem.MolFromSmiles(smiles)
#     mol = Chem.AddHs(mol)
    
#     # Generate 3D coordinates
#     AllChem.EmbedMolecule(mol)
#     AllChem.UFFOptimizeMolecule(mol)

#     # Create 3D visualization using Py3Dmol
#     mol_block = Chem.MolToMolBlock(mol)
#     viewer = py3Dmol.view(width=1920, height=1920)
#     viewer.addModel(mol_block, 'mol')
#     viewer.setStyle({'stick': {}})
#     viewer.setBackgroundColor('white')
#     viewer.zoomTo()
#     png_data = viewer.png()
#     img = Image.open(io.BytesIO(png_data))
#     img.save(img_name)
    
# def smiles_to_3d(smiles_list, output_dir):
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
        
#     for idx, smiles in enumerate(smiles_list):
#         mol = Chem.MolFromSmiles(smiles)
#         mol = Chem.AddHs(mol)  # Add hydrogens to the molecule
#         AllChem.EmbedMolecule(mol)  # Generate 3D coordinates
#         AllChem.UFFOptimizeMolecule(mol)  # Optimize geometry
        
#         # Generate 3D structure in mol block format
#         mb = Chem.MolToMolBlock(mol)
        
#         # Visualize with py3Dmol using stick representation
#         viewer = py3Dmol.view(width=500, height=500)
#         viewer.addModel(mb, "mol")  # Add the 3D molecule
#         viewer.setStyle({'stick': {}})  # Use stick style
#         viewer.zoomTo()
#         png_filename = os.path.join(output_dir, f'molecule_{idx}.png')
#         # Get PNG binary data from py3Dmol
#         png_binary = viewer.png()  # Capture PNG as binary data

#         # Save the binary data as a PNG file
#         png_filename = os.path.join(output_dir, f'molecule_{idx}.png')
#         with open(png_filename, 'wb') as f:
#             f.write(png_binary)
        
# smiles_to_3d(filt, '/mnt/ssd_raid/workspace-jjaramillo/PDAA/stages/imgs/')
    
# for i, smiles in enumerate(filt):
#     smiles_to_3d_image(smiles, f"/mnt/ssd_raid/workspace-jjaramillo/PDAA/stages/imgs/mol_{i}.png")
 
# for i, smiles in enumerate(filt):
#     mol = Chem.MolFromSmiles(smiles)
#     img = Draw.MolToImage(mol, size=img_size)
#     img.save(f"/mnt/ssd_raid/workspace-jjaramillo/PDAA/stages/imgs/mol_{i}.png")


def smiles_to_png(smiles, filename):
  mol = Chem.MolFromSmiles(smiles)
  img = Draw.MolToImage(mol, size=(1920, 1920))
  img.save(filename)

df = list(pd.read_parquet('/mnt/ssd_raid/workspace-jjaramillo/PDAA/raw/zinc_phthalates/zinc1.parquet').head(20)['smiles'])
for i, smi in enumerate(df):
   smiles_to_png(smi, '/mnt/ssd_raid/workspace-jjaramillo/PDAA/stages/imgs/m{}.png'.format(i))
    
image_list = []
for filename in sorted(glob.glob("/mnt/ssd_raid/workspace-jjaramillo/PDAA/stages/imgs/*.png")):  # Load images in order
    img = Image.open(filename)
    image_list.append(img)
    
image_list[0].save("molecules2.gif", save_all=True, append_images=image_list[1:], duration=200, loop=0)
