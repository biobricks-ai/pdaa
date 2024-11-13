import pandas as pd
from rdkit import Chem
from tqdm import tqdm
from multiprocessing import Pool
import os
import glob
import requests
import pubchempy as pcp

def cleanGeneration(generation):
    cleansed=[]
    for generated in tqdm(generation):
        mol = Chem.MolFromSmiles(generated['SMILES'])
        if mol is not None:
            canon = Chem.MolToSmiles(mol)
            results = pcp.get_compounds(canon, namespace="smiles")
            if results[0].cid is None:
                cleansed.append(generated)
            else:
                print('Excluding {} compound. Found on pubchem.'.format(generated['SGXID']))
                
    return cleansed

def get_casrn_from_smiles(smiles):
    # Convert SMILES to PubChem CID
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/cids/JSON"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        cid = data.get('IdentifierList', {}).get('CID', [None])[0]
        if cid:
            # Retrieve CASRN using CID
            cas_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/xrefs/RegistryID/JSON"
            cas_response = requests.get(cas_url)
            if cas_response.status_code == 200:
                cas_data = cas_response.json()
                casrn = cas_data.get('InformationList', {}).get('Information', [{}])[0].get('RegistryID', None)
                return casrn
    return None

if __name__ == "__main__":

    os.makedirs('raw/CTD_MESH_phthalates', exist_ok=True)
    file_paths = glob.glob('raw/zinc_phthalates/*.parquet')
    print(file_paths)
    df = pd.read_parquet(file_paths)
    df_no_duplicates = df.drop_duplicates(subset='smiles')
    for smi in df_no_duplicates['smiles']:
        print(get_casrn_from_smiles(smi))
        # cas=[]
        # mol = Chem.MolFromSmiles(smi)
        # if mol is not None:
        #     canon = Chem.MolToSmiles(mol)
        #     results = pcp.get_compounds(canon, namespace="smiles")
        #     if results[0].cid is None:
        #         print(dir(results[0]))
        #         input()