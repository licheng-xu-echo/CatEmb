from rdkit import Chem,RDLogger
from collections import Counter
import os,logging,sys,argparse
from datetime import datetime
import random
import numpy as np
import torch
pt = Chem.GetPeriodicTable()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  
    
def str_to_bool(value):
    if value.lower() in {"true", "yes", "1"}:
        return True
    elif value.lower() in {"false", "no", "0"}:
        return False
    else:
        raise argparse.ArgumentTypeError("must be True/False")
    
def calculate_gaussview_spin_multiplicity(element_list, charge=0):
    
    if isinstance(element_list[0],str):
        element_list = [pt.GetAtomicNumber(element) for element in element_list]
    
    if not element_list:
        return 1  # 

    element_counts = Counter(element_list)
    total_valence_electrons = 0
    for element, count in element_counts.items():
        
        total_valence_electrons += pt.GetNOuterElecs(element) * count
    total_electrons_in_system = total_valence_electrons - charge

    if total_electrons_in_system % 2 == 0:
        return 1
    else:
        return 2
    
def symbol_pos_to_xyz_file(symbols,positions,xyz_file,title=""):
    with open(xyz_file, 'w') as f:
        f.write(f"{len(symbols)}\n")
        f.write(f"{title}\n")
        for symbol, pos in zip(symbols, positions):
            f.write(f"{symbol:4s} {pos[0]:15f} {pos[1]:15f} {pos[2]:15f}\n")
            
def setup_logger(save_dir):
    RDLogger.DisableLog("rdApp.*")
    RDLogger.DisableLog("rdApp.warning")
    os.makedirs(save_dir, exist_ok=True)
    #os.makedirs(f"{config.model.save_dir}/{config.data.data_path.split('/')[-1]}", exist_ok=True)
    dt = datetime.strftime(datetime.now(), "%y%m%d-%H%Mh")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(f"{save_dir}/{dt}.log")
    sh = logging.StreamHandler(sys.stdout)
    fh.setLevel(logging.INFO)
    sh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger

def link_lig_to_metal(lig_mol,metal_type,coord_at_idx_lst,metal_chrg):
    rw_lig_mol = Chem.RWMol(lig_mol)
    rw_lig_mol.UpdatePropertyCache(strict=False)
    metal_idx = rw_lig_mol.AddAtom(Chem.Atom(metal_type))
    for at_idx in coord_at_idx_lst:
        rw_lig_mol.AddBond(at_idx,metal_idx,Chem.BondType.DATIVE)
    rw_lig_mol.GetAtomWithIdx(metal_idx).SetFormalCharge(metal_chrg)
    rw_lig_mol.UpdatePropertyCache(strict=True)
    cat_mol = rw_lig_mol.GetMol()
    return cat_mol

def link_newlig_to_cat(lig_mol,cat_mol,metal_type,coord_at_idx_lst,metal_at_idx=None):
    if metal_at_idx is None:
        metal_at_idx = cat_mol.GetNumAtoms() - 1
    assert cat_mol.GetAtomWithIdx(metal_at_idx).GetSymbol() == metal_type
    rw_cat_mol = Chem.RWMol(cat_mol)
    rw_cat_mol.UpdatePropertyCache(strict=False)

    rw_lig_mol = Chem.RWMol(lig_mol)
    rw_lig_mol.UpdatePropertyCache(strict=False)

    rw_cat_mol.InsertMol(rw_lig_mol)

    for at_idx in coord_at_idx_lst:
        rw_cat_mol.AddBond(at_idx+cat_mol.GetNumAtoms(),metal_at_idx,Chem.BondType.DATIVE)
    rw_cat_mol.UpdatePropertyCache(strict=True)
    cat_mol = rw_cat_mol.GetMol()
    return cat_mol,metal_at_idx