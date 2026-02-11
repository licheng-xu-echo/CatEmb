
'''
import glob
import pandas as pd
import numpy as np
from rdkit import Chem,RDLogger
from tqdm import tqdm
from rdkit.Chem import AllChem
RDLogger.DisableLog('rdApp.*')
pt = Chem.GetPeriodicTable()
random_seed = 42

def process_data(smiles_lst,process_blk=False):
    dataset = []
    split_smiles = []
    for smiles in tqdm(smiles_lst):
        try:
            if '.' in smiles:
                if process_blk:
                    smi_blks = sorted(smiles.split('.'),key=lambda x:len(x))
                    lig_blk = smi_blks[-1]
                    
                    split_smiles.append(smiles)
                    smiles = lig_blk
                    print(f'ligand block: {smiles}')
                else:
                    split_smiles.append(smiles)
                    continue
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            if mol.GetNumAtoms() == 1:
                continue
            mol_H = AllChem.AddHs(mol)
            flag = AllChem.EmbedMolecule(mol_H, randomSeed=random_seed)
            if flag != 0:
                continue
            pos = mol_H.GetConformer().GetPositions()
            atoms = [atom.GetAtomicNum() for atom in mol_H.GetAtoms()]
            data = {'smiles':Chem.MolToSmiles(mol), 'atoms':atoms, 'pos':pos}
            dataset.append(data)
        except:
            pass
    return dataset, split_smiles

def link_lig_to_metal(lig_mol,metal_type,coord_at_idx_lst):
    rw_lig_mol = Chem.RWMol(lig_mol)
    rw_lig_mol.UpdatePropertyCache(strict=False)
    metal_idx = rw_lig_mol.AddAtom(Chem.Atom(metal_type))
    for at_idx in coord_at_idx_lst:
        rw_lig_mol.AddBond(at_idx,metal_idx,Chem.BondType.DATIVE)
    rw_lig_mol.UpdatePropertyCache(strict=True)
    cat_mol = rw_lig_mol.GetMol()
    return cat_mol

metal_lst = ['Mg', 'Sc', 'Ti', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Y', 'Ru', 'Rh', 'Pd', 'Ag', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Os', 'Ir', 'Pt', 'Au']

smiles_geom_info = np.load('../dataset/processed/original_smiles.npy',allow_pickle=True).item()


### SadPhos
print("SadPhos")
sadphos_smarts_idx_map = {'O=[SX3]([*:4])[N]([*:3])[C]([H])([*:1])([*:2]).[PX3]([#6])([#6])':[9,1],          # P, S
                      '[*][SX3](N([*])[CX4;!R]([*])C1=CC=CC=C1[PX3]([#6])([#6]))=O': [12,1],
                      '[*][CX4;!R]([CX4;!R][PX3]([#6])[#6])N([*])[S@@X3]([*])=O': [3,8],
                      '[*]C1([*])C2=C(C([CX4;!R]([*])N(*)[S@@X3]([*])=O)=CC=C2)OC3=C([PX3]([#6])([#6]))C=CC=C31':[19, 10]}

cat_dataset = []
for smiles in tqdm(smiles_geom_info.keys()):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    for smarts in sadphos_smarts_idx_map.keys(): ###
        smarts_mol = Chem.MolFromSmarts(smarts)
        idx_lst = sadphos_smarts_idx_map[smarts] ###
        if mol.HasSubstructMatch(smarts_mol):
            match_at_idx_lst = mol.GetSubstructMatch(smarts_mol)
            link_atom_idx = [match_at_idx_lst[idx] for idx in idx_lst]
            check = True
            for atom_idx in link_atom_idx:
                atom_symbol = mol.GetAtomWithIdx(atom_idx).GetSymbol()
                if atom_symbol not in ['P','S']:
                    check = False
                    break
            if not check:
                continue
            for metal in metal_lst:
                cat_mol = link_lig_to_metal(mol,metal,link_atom_idx)
                cat_smi = Chem.MolToSmiles(Chem.RemoveHs(cat_mol))
                flag = AllChem.EmbedMolecule(cat_mol,randomSeed=random_seed)
                if flag == 0:
                    cat_dataset.append({'cat_smiles':cat_smi,'lig_smiles':smiles,
                                        'atoms':[atom.GetAtomicNum() for atom in cat_mol.GetAtoms()], 
                                        'pos':cat_mol.GetConformer().GetPositions()})
                    

np.save("../dataset/processed/sadphos_complex.npy",cat_dataset)


### PHOX
print("PHOX")
phox_smarts_idx_map = {'P-[#6,#7]~[#6]-[C;H0]1=N[C,c][C,c]O1':[0,4]}

cat_dataset = []
for smiles in tqdm(smiles_geom_info.keys()):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    for smarts in phox_smarts_idx_map.keys(): ####
        smarts_mol = Chem.MolFromSmarts(smarts)
        idx_lst = phox_smarts_idx_map[smarts] ####
        if mol.HasSubstructMatch(smarts_mol):
            match_at_idx_lst = mol.GetSubstructMatch(smarts_mol)
            link_atom_idx = [match_at_idx_lst[idx] for idx in idx_lst]
            check = True
            for atom_idx in link_atom_idx:
                atom_symbol = mol.GetAtomWithIdx(atom_idx).GetSymbol()
                if atom_symbol not in ['P','N']:
                    check = False
                    print(123)
                    break
            if not check:
                continue
            for metal in metal_lst:
                cat_mol = link_lig_to_metal(mol,metal,link_atom_idx)
                cat_smi = Chem.MolToSmiles(Chem.RemoveHs(cat_mol))
                flag = AllChem.EmbedMolecule(cat_mol,randomSeed=random_seed)
                if flag == 0:
                    cat_dataset.append({'cat_smiles':cat_smi,'lig_smiles':smiles,
                                        'atoms':[atom.GetAtomicNum() for atom in cat_mol.GetAtoms()], 
                                        'pos':cat_mol.GetConformer().GetPositions()})

np.save("../dataset/processed/phox_complex.npy",cat_dataset)



### BOX
print("BOX")
box_smarts_idx_map = {'O1[C,c][C,c]N=[C]1~[#6]~[C]2=N[C,c][C,c]O2':[3,7]}

cat_dataset = []
for smiles in tqdm(smiles_geom_info.keys()):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    for smarts in box_smarts_idx_map.keys(): ####
        smarts_mol = Chem.MolFromSmarts(smarts)
        idx_lst = box_smarts_idx_map[smarts] ####
        if mol.HasSubstructMatch(smarts_mol):
            match_at_idx_lst = mol.GetSubstructMatch(smarts_mol)
            link_atom_idx = [match_at_idx_lst[idx] for idx in idx_lst]
            check = True
            for atom_idx in link_atom_idx:
                atom_symbol = mol.GetAtomWithIdx(atom_idx).GetSymbol()
                if atom_symbol not in ['N']:
                    check = False
                    print(123)
                    break
            if not check:
                continue
            for metal in metal_lst:
                cat_mol = link_lig_to_metal(mol,metal,link_atom_idx)
                cat_smi = Chem.MolToSmiles(Chem.RemoveHs(cat_mol))
                flag = AllChem.EmbedMolecule(cat_mol,randomSeed=random_seed)
                if flag == 0:
                    cat_dataset.append({'cat_smiles':cat_smi,'lig_smiles':smiles,
                                        'atoms':[atom.GetAtomicNum() for atom in cat_mol.GetAtoms()], 
                                        'pos':cat_mol.GetConformer().GetPositions()})

np.save("../dataset/processed/box_complex.npy",cat_dataset)


### Salen
print("Salen")
salen_smarts_idx_map = {'O-c1ccccc1-C=[N,n]~[#6]~[#6]~[N,n]=C-c2ccccc2O':[0,8,11,19]}

cat_dataset = []
for smiles in tqdm(smiles_geom_info.keys()):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    for smarts in salen_smarts_idx_map.keys(): ####
        smarts_mol = Chem.MolFromSmarts(smarts)
        idx_lst = salen_smarts_idx_map[smarts] ####
        if mol.HasSubstructMatch(smarts_mol):
            match_at_idx_lst = mol.GetSubstructMatch(smarts_mol)
            link_atom_idx = [match_at_idx_lst[idx] for idx in idx_lst]
            check = True
            for atom_idx in link_atom_idx:
                atom_symbol = mol.GetAtomWithIdx(atom_idx).GetSymbol()
                if atom_symbol not in ['N','O']: ###
                    check = False
                    print(123)
                    break
            if not check:
                continue
            for metal in metal_lst:
                cat_mol = link_lig_to_metal(mol,metal,link_atom_idx)
                cat_smi = Chem.MolToSmiles(Chem.RemoveHs(cat_mol))
                flag = AllChem.EmbedMolecule(cat_mol,randomSeed=random_seed)
                if flag == 0:
                    cat_dataset.append({'cat_smiles':cat_smi,'lig_smiles':smiles,
                                        'atoms':[atom.GetAtomicNum() for atom in cat_mol.GetAtoms()], 
                                        'pos':cat_mol.GetConformer().GetPositions()})

np.save("../dataset/processed/salen_complex.npy",cat_dataset) ###


Chem.MolFromSmarts('O-c1ccccc1-C=[N,n]~[#6]~[#6]~[N,n]=C-c2ccccc2O')



### Bisphosphine
print("Bisphosphine")
bisphos_smarts_idx_map = {'[P]~[#6,c]~[P]':[0,2],
 '[P]~[#6,c]~[#6,c]~[P]':[0,3],
 '[P]~[#6,c]~[#6,c]~[#6,c]~[P]':[0,4],
 '[P]~[#6,c]~[#6,c]~[#6,c]~[#6,c]~[P]':[0,5],
 'P-c1c2ccccc2ccc1-c3c(P)ccc4ccccc34':[0,13],
 'PC12C3C4C5C1[Fe]25436789C%10C6C7C8C%109P':[0,12],
 '[P;H0](-[#6])([-#6])-[#6]~[#6]~[P;H0](-[#6])-[#6]':[0,5],
 }

cat_dataset = []
for smiles in tqdm(smiles_geom_info.keys()):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    for smarts in bisphos_smarts_idx_map.keys(): ####
        smarts_mol = Chem.MolFromSmarts(smarts)
        idx_lst = bisphos_smarts_idx_map[smarts] ####
        if mol.HasSubstructMatch(smarts_mol):
            match_at_idx_lst = mol.GetSubstructMatch(smarts_mol)
            link_atom_idx = [match_at_idx_lst[idx] for idx in idx_lst]
            check = True
            for atom_idx in link_atom_idx:
                atom_symbol = mol.GetAtomWithIdx(atom_idx).GetSymbol()
                if atom_symbol not in ['P']: ###
                    check = False
                    print(123)
                    break
            if not check:
                continue
            for metal in metal_lst:
                cat_mol = link_lig_to_metal(mol,metal,link_atom_idx)
                cat_smi = Chem.MolToSmiles(Chem.RemoveHs(cat_mol))
                flag = AllChem.EmbedMolecule(cat_mol,randomSeed=random_seed)
                if flag == 0:
                    cat_dataset.append({'cat_smiles':cat_smi,'lig_smiles':smiles,
                                        'atoms':[atom.GetAtomicNum() for atom in cat_mol.GetAtoms()], 
                                        'pos':cat_mol.GetConformer().GetPositions()})

np.save("../dataset/processed/bisphos_complex.npy",cat_dataset) ###
Chem.MolFromSmarts('O-c1ccccc1-C=[N,n]~[#6]~[#6]~[N,n]=C-c2ccccc2O')



### NHC
print("NHC")
nhc_smarts_idx_map = {'[n,N,n+,N+]1~[c,C;H1]~[n,N,n+,N+]~[c,C]~[c,C]1':[1]}
cat_dataset = []
for smiles in tqdm(smiles_geom_info.keys()):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    for smarts in nhc_smarts_idx_map.keys(): ####
        smarts_mol = Chem.MolFromSmarts(smarts)
        idx_lst = nhc_smarts_idx_map[smarts] ####
        if mol.HasSubstructMatch(smarts_mol):
            match_at_idx_lst = mol.GetSubstructMatch(smarts_mol)
            link_atom_idx = [match_at_idx_lst[idx] for idx in idx_lst]
            check = True
            for atom_idx in link_atom_idx:
                atom_symbol = mol.GetAtomWithIdx(atom_idx).GetSymbol()
                if atom_symbol not in ['C']: ###
                    check = False
                    print(123)
                    break
            if not check:
                continue
            for metal in metal_lst:
                cat_mol = link_lig_to_metal(mol,metal,link_atom_idx)
                cat_smi = Chem.MolToSmiles(Chem.RemoveHs(cat_mol))
                flag = AllChem.EmbedMolecule(cat_mol,randomSeed=random_seed)
                if flag == 0:
                    cat_dataset.append({'cat_smiles':cat_smi,'lig_smiles':smiles,
                                        'atoms':[atom.GetAtomicNum() for atom in cat_mol.GetAtoms()], 
                                        'pos':cat_mol.GetConformer().GetPositions()})

np.save("../dataset/processed/nhc_complex.npy",cat_dataset) ###




### Amino Acid
print("Amino Acid")
anion_amino_acid_idx_lst = [0,3]
neutral_amino_acid_smiles = '[N][CX4][C](=[O])[O;H1]'
anion_amino_acid_smarts = '[N][CX4][C](=[O])[O;H0-]'
dehydrogen_rxn_template = "[N:5][CX4:1][C:2](=[O:3])[O;H1:4]>>[N:5][CX4:1][C:2](=[O:3])[O-1:4]"
neutral_amino_acid_mol = Chem.MolFromSmarts(neutral_amino_acid_smiles)
anion_amino_acid_mol = Chem.MolFromSmarts(anion_amino_acid_smarts)
rxn = AllChem.ReactionFromSmarts(dehydrogen_rxn_template)

cat_dataset = []
cat_mol_lst = []
for smiles in tqdm(smiles_geom_info.keys()):
    mol = Chem.MolFromSmiles(smiles)
    
    
    if mol.HasSubstructMatch(neutral_amino_acid_mol):
        pdt_mols = rxn.RunReactants((mol,))
        mol = pdt_mols[0][0]
    elif mol.HasSubstructMatch(anion_amino_acid_mol):
        mol = mol
    else:
        continue
    mol = Chem.AddHs(mol)
    match_at_idx_lst = mol.GetSubstructMatch(anion_amino_acid_mol)
    link_atom_idx = [match_at_idx_lst[idx] for idx in anion_amino_acid_idx_lst]
    check = True
    for atom_idx in link_atom_idx:
        atom_symbol = mol.GetAtomWithIdx(atom_idx).GetSymbol()
        if atom_symbol not in ['N','O']:
            check = False
            print(123)
            break
    if not check:
        continue
    for metal in metal_lst:
        cat_mol = link_lig_to_metal(mol,metal,link_atom_idx)
        cat_smi = Chem.MolToSmiles(Chem.RemoveHs(cat_mol))
        flag = AllChem.EmbedMolecule(cat_mol,randomSeed=random_seed)
        
        if flag == 0:
            cat_dataset.append({'cat_smiles':cat_smi,'lig_smiles':smiles,
                                'atoms':[atom.GetAtomicNum() for atom in cat_mol.GetAtoms()], 
                                'pos':cat_mol.GetConformer().GetPositions()})
            cat_mol_lst.append(cat_mol)
    

np.save("../dataset/processed/amino_acid_complex.npy",cat_dataset) ###



### N,N'-Dioxide
print("N,N'-Dioxide")
nn_dioxide_smarts_idx_map = {'[O-][N+]1(CCCCC1C=O)CCC[N+]2(C(C=O)CCCC2)[O-]':[0,8,15,20],
                          '[O-][N+]1(CCC[N+]2([O-])CCCC2C=O)CCCC1C=O':[0,6,12,18]}

cat_dataset = []
for smiles in tqdm(smiles_geom_info.keys()):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    for smarts in nn_dioxide_smarts_idx_map.keys(): ####
        smarts_mol = Chem.MolFromSmarts(smarts)
        idx_lst = nn_dioxide_smarts_idx_map[smarts] ####
        if mol.HasSubstructMatch(smarts_mol):
            match_at_idx_lst = mol.GetSubstructMatch(smarts_mol)
            link_atom_idx = [match_at_idx_lst[idx] for idx in idx_lst]
            check = True
            for atom_idx in link_atom_idx:
                atom_symbol = mol.GetAtomWithIdx(atom_idx).GetSymbol()
                if atom_symbol not in ['O']: ###
                    check = False
                    print(123)
                    break
            if not check:
                continue
            for metal in metal_lst:
                cat_mol = link_lig_to_metal(mol,metal,link_atom_idx)
                cat_smi = Chem.MolToSmiles(Chem.RemoveHs(cat_mol))
                flag = AllChem.EmbedMolecule(cat_mol,randomSeed=random_seed)
                if flag == 0:
                    cat_dataset.append({'cat_smiles':cat_smi,'lig_smiles':smiles,
                                        'atoms':[atom.GetAtomicNum() for atom in cat_mol.GetAtoms()], 
                                        'pos':cat_mol.GetConformer().GetPositions()})

np.save("../dataset/processed/nn_dioxide_complex.npy",cat_dataset) ###




### Monophosphine
monophos_smarts_idx_map = {'[P;X3;H0;v3;!$(P=[O,S,N])](-[#6,c])(-[#6,c])-[#6,c]':[0],
                          }
cat_dataset = []
for smiles in tqdm(smiles_geom_info.keys()):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    for smarts in monophos_smarts_idx_map.keys(): ####
        smarts_mol = Chem.MolFromSmarts(smarts)
        idx_lst = monophos_smarts_idx_map[smarts] ####
        if mol.HasSubstructMatch(smarts_mol):
            match_at_idx_lst = mol.GetSubstructMatch(smarts_mol)
            if len(mol.GetSubstructMatches(smarts_mol)) > 1:
                continue
            link_atom_idx = [match_at_idx_lst[idx] for idx in idx_lst]
            check = True
            for atom_idx in link_atom_idx:
                atom_symbol = mol.GetAtomWithIdx(atom_idx).GetSymbol()
                if atom_symbol not in ['P']: ###
                    check = False
                    print(123)
                    break
            if not check:
                continue
            for metal in metal_lst:
                cat_mol = link_lig_to_metal(mol,metal,link_atom_idx)
                cat_smi = Chem.MolToSmiles(Chem.RemoveHs(cat_mol))
                flag = AllChem.EmbedMolecule(cat_mol,randomSeed=random_seed)
                if flag == 0:
                    cat_dataset.append({'cat_smiles':cat_smi,'lig_smiles':smiles,
                                        'atoms':[atom.GetAtomicNum() for atom in cat_mol.GetAtoms()], 
                                        'pos':cat_mol.GetConformer().GetPositions()})

np.save("../dataset/processed/monophos_complex.npy",cat_dataset) ###
'''

from catemb.data import CatDataset
random_seed = 42
new_dataset = CatDataset(root="../dataset/processed",name="lig_cat_dataset",trunc=0,seed=random_seed,save_smiles=True)