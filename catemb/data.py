import torch,warnings,logging,os
import numpy as np
from torch_geometric.data import InMemoryDataset,Data
from torch_geometric.loader import DataLoader
from rdkit import Chem,RDLogger
from torch_geometric.data.separate import separate
from sklearn.utils import shuffle
from rdkit.Chem import AllChem
from multiprocessing import Pool, cpu_count
from .utils import calculate_gaussview_spin_multiplicity
from tqdm import tqdm
warnings.filterwarnings("ignore")
RDLogger.DisableLog('rdApp.*')
pt = Chem.GetPeriodicTable()

NUM_ATOM_TYPE = 90
NUM_DEGRESS_TYPE = 11
NUM_FORMCHRG_TYPE = 9           # -4,-3,-2,-2,-1,0,1,2,3,4
NUM_HYBRIDTYPE = 6
NUM_CHIRAL_TYPE = 3
NUM_AROMATIC_NUM = 2
NUM_VALENCE_TYPE = 7
NUM_Hs_TYPE = 5
NUM_RS_TPYE = 3
NUM_RADICAL_TYPES = 5


NUM_BOND_INRING = 2
NUM_BOND_ISCONJ = 2
ATOM_FEAT_DIMS = [NUM_ATOM_TYPE,NUM_DEGRESS_TYPE,NUM_FORMCHRG_TYPE,NUM_HYBRIDTYPE,NUM_CHIRAL_TYPE,
                    NUM_AROMATIC_NUM,NUM_VALENCE_TYPE,NUM_Hs_TYPE,NUM_RS_TPYE]
ATOM_LST = [pt.GetElementSymbol(i) for i in range(1,NUM_ATOM_TYPE+1)]

ATOM_DICT = {symbol: i for i, symbol in enumerate(ATOM_LST)}
MAX_NEIGHBORS = 10
CHIRAL_TAG_LST = [Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
                  Chem.rdchem.ChiralType.CHI_UNSPECIFIED]
CHIRAL_TAG_DICT = {ct: i for i, ct in enumerate(CHIRAL_TAG_LST)}
HYBRIDTYPE_LST = [Chem.rdchem.HybridizationType.SP,Chem.rdchem.HybridizationType.SP2,Chem.rdchem.HybridizationType.SP3,
                  Chem.rdchem.HybridizationType.SP3D,Chem.rdchem.HybridizationType.SP3D2,Chem.rdchem.HybridizationType.UNSPECIFIED]
HYBRIDTYPE_DICT = {hb: i for i, hb in enumerate(HYBRIDTYPE_LST)}
VALENCE_LST = [0, 1, 2, 3, 4, 5, 6]
VALENCE_DICT = {vl: i for i, vl in enumerate(VALENCE_LST)}
NUM_Hs_LST = [0, 1, 3, 4, 5]
NUM_Hs_DICT = {nH: i for i, nH in enumerate(NUM_Hs_LST)}
BOND_TYPE_LST = [Chem.rdchem.BondType.SINGLE,
                 Chem.rdchem.BondType.DOUBLE,
                 Chem.rdchem.BondType.TRIPLE,
                 Chem.rdchem.BondType.AROMATIC,
                 Chem.rdchem.BondType.DATIVE,
                 Chem.rdchem.BondType.UNSPECIFIED,
                 ]
NUM_BOND_TYPE = len(BOND_TYPE_LST)
BOND_DIR_LST = [ # only for double bond stereo information
                Chem.rdchem.BondDir.NONE,
                Chem.rdchem.BondDir.ENDUPRIGHT,
                Chem.rdchem.BondDir.ENDDOWNRIGHT,
                Chem.rdchem.BondDir.BEGINDASH,
                Chem.rdchem.BondDir.BEGINWEDGE,
                Chem.rdchem.BondDir.EITHERDOUBLE]
NUM_BOND_DIRECTION = len(BOND_DIR_LST)
BOND_STEREO_LST = [Chem.rdchem.BondStereo.STEREONONE,
                   Chem.rdchem.BondStereo.STEREOE,
                   Chem.rdchem.BondStereo.STEREOZ,
                   Chem.rdchem.BondStereo.STEREOANY,
                   Chem.rdchem.BondStereo.STEREOATROPCW,
                   Chem.rdchem.BondStereo.STEREOATROPCCW,
                   ]
NUM_BOND_STEREO = len(BOND_STEREO_LST)
BOND_FEAT_DIME = [NUM_BOND_TYPE,NUM_BOND_DIRECTION,NUM_BOND_STEREO,NUM_BOND_INRING,NUM_BOND_ISCONJ]
FORMAL_CHARGE_LST = [-1, -2, 1, 2, 0]
FC_DICT = {fc: i for i, fc in enumerate(FORMAL_CHARGE_LST)}
RS_TAG_LST = ["R","S","None"]
RS_TAG_DICT = {rs: i for i, rs in enumerate(RS_TAG_LST)}

element_energy_map = {'H': -0.393482763936,
 'Li': -0.180071686575,
 'B': -0.952436614164,
 'C': -1.795110518041,
 'N': -2.60945245463,
 'O': -3.769421097051,
 'F': -4.619339964238,
 'Mg': -0.465974663792,
 'Al': -0.905328611479,
 'Si': -1.571424085131,
 'P': -2.377807088084,
 'S': -3.148271017078,
 'Cl': -4.482525134961,
 'Sc': -0.854183293246,
 'Ti': -1.367057306084,
 'V': -1.718125190805,
 'Cr': -1.747497514035,
 'Mn': -0.182819183551,
 'Fe': -2.946795476061,
 'Co': -3.506789849217,
 'Ni': -4.668327435364,
 'Cu': -3.748006130985,
 'Zn': -0.527521402296,
 'Ga': -1.081111835714,
 'Ge': -1.809903783921,
 'As': -2.239425948594,
 'Se': -3.120436197255,
 'Br': -4.048339371234,
 'Y': -1.194852897131,
 'Zr': -1.310653757662,
 'Nb': -1.781200839604,
 'Mo': -1.784620418901,
 'Ru': -2.439277446685,
 'Rh': -3.895815471415,
 'Pd': -4.40984529993,
 'Ag': -3.821738210271,
 'Cd': -0.533037255301,
 'In': -1.12593777889,
 'Sn': -2.012896616134,
 'Sb': -2.164228788033,
 'Te': -3.009090964572,
 'I': -3.77963026339,
 'Ba': -0.433642020732,
 'La': -1.20479302375,
 'Ce': -0.89950888591,
 'Pr': -0.893369289912,
 'Nd': -0.887229657264,
 'Sm': -0.874950428917,
 'Eu': -0.868810832956,
 'Gd': -0.862671273816,
 'Tb': -0.856531621116,
 'Dy': -0.850392062002,
 'Ho': -0.84425246614,
 'Er': -0.83811283353,
 'Tm': -0.831973237668,
 'Yb': -0.825833568309,
 'Lu': -0.819694009197,
 'Hf': -1.311998765755,
 'Ta': -1.90481355056,
 'W': -2.215077772613,
 'Re': -0.573920486119,
 'Os': -1.763844412027,
 'Ir': -0.641964304634,
 'Pt': -4.437458572161,
 'Au': -3.802619448068,
 'Hg': -0.848032246708,
 'Tl': -1.438685144905,
 'Pb': -2.204840980245,
 'Bi': -2.26665341636}

def mol2graphinfo(mol,chrg,multi):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices
    """
    assert chrg in [-4,-3,-2,-1,0,1,2,3,4] and multi in [1,2,3,4,5], f"Charges ({chrg}) and mults ({multi}) not in [-2,-1,0,1,2,3,4] and [1,2,3,4,5] respectively"
    atom_features_list = []
    atom_mass_list = []
    
    for atom in mol.GetAtoms():
        atom_feature = [ATOM_DICT.get(atom.GetSymbol()),
                        min(atom.GetDegree(),MAX_NEIGHBORS),
                        chrg+4,                                         # -4,-3,-2,-1,0,1,2,3,4
                        HYBRIDTYPE_DICT.get(atom.GetHybridization(), 5),
                        CHIRAL_TAG_DICT.get(atom.GetChiralTag(),2),
                        int(atom.GetIsAromatic()),
                        VALENCE_DICT.get(atom.GetTotalValence(), 6),
                        NUM_Hs_DICT.get(atom.GetTotalNumHs(), 4),
                        RS_TAG_DICT.get(atom.GetPropsAsDict().get("_CIPCode", "None"), 2),
                        multi-1]
        atom_mass = atom.GetMass()
        atom_features_list.append(atom_feature)
        atom_mass_list.append(atom_mass)
    x = torch.tensor(np.array(atom_features_list),dtype=torch.long)
    atom_mass = torch.from_numpy(np.array(atom_mass_list))
    # bonds
    num_bond_features = 5   # bond type, bond direction, bond stereo, isinring, isconjugated
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [BOND_TYPE_LST.index(bond.GetBondType()),
                            BOND_DIR_LST.index(bond.GetBondDir()),
                            BOND_STEREO_LST.index(bond.GetStereo()),
                            int(bond.IsInRing()),
                            int(bond.GetIsConjugated())]

            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)
        edge_index = np.array(edges_list).T
        
        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)

    else:   # mol has no bonds
        edge_index = np.empty((2,0),dtype=np.int32)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    edge_index = torch.tensor(edge_index, dtype=torch.long)
    return x,edge_index,edge_attr,atom_mass

def generate_custom_list(intervals):
    result = []
    for idx, length in enumerate(intervals):
        result.extend([idx] * length)
    return result


def get_idx_split(data_size, train_size, valid_size, seed):
    ids = shuffle(range(data_size), random_state=seed)
    if abs(train_size + valid_size - data_size) < 2:
        train_idx, val_idx = torch.tensor(ids[:train_size]), torch.tensor(ids[train_size:])
        test_idx = val_idx
    else:    
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
    split_dict = {'train':train_idx, 'valid':val_idx, 'test':test_idx}
    return split_dict


def process_single_mol(args):
    idx, _data, seed, save_smi, read_coord = args
    smiles = _data['smiles']
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    mol = AllChem.AddHs(mol)
    
    
    atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    atom_formal_chrgs = [atom.GetFormalCharge() for atom in mol.GetAtoms()]
    mol_formal_chrgs = sum(atom_formal_chrgs)
    mult = calculate_gaussview_spin_multiplicity(atom_symbols,mol_formal_chrgs)
    atom_feat, edge_index, edge_attr, atom_mass = mol2graphinfo(mol, mol_formal_chrgs, mult)
    if not read_coord:
        flag = AllChem.EmbedMolecule(mol, randomSeed=seed)
        if flag != 0:
            print(f'[WARNING] {smiles} failed to embed')
            return None
        pos = torch.from_numpy(mol.GetConformer().GetPositions()).float()
        E = torch.tensor([0.])
    else:
        pos = torch.from_numpy(_data['xtbopt_coords']).float()
        E = torch.tensor([_data['xtb_energy']])
        atom_sym_lst = [pt.GetElementSymbol(int(item[0]+1)) for item in atom_feat]
        atom_e_lst = [element_energy_map[at_sym] for at_sym in atom_sym_lst]
        E = E - sum(atom_e_lst)
    
    if not save_smi:
        data_ = Data(
            x=atom_feat,
            edge_index=edge_index,
            edge_attr=edge_attr,
            mol_coords=pos,
            idx=idx,
            E=E
        )
    else:
        data_ = Data(
            x=atom_feat,
            edge_index=edge_index,
            edge_attr=edge_attr,
            mol_coords=pos,
            idx=idx,
            smiles=smiles,
            E=E
        )
    return data_

class CatEmbDataset(InMemoryDataset):
    pass

class CatDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None, seed=42, trunc=0, num_workers=1, save_smiles=False, read_coord=False, **kwargs):
        self.root = root
        self.name = name
        self.datafile = f'{root}/{name}.npy'
        self.seed = seed
        self.trunc = trunc
        self.num_workers = num_workers if num_workers else cpu_count()
        self.save_smiles = save_smiles
        self.read_coord = read_coord
        logging.info(f'[INFO] Using {self.num_workers} cores for parallel processing.')
        super().__init__(root, transform, pre_transform)
        
        data, slices = torch.load(self.processed_paths[0], weights_only=False)
        self.data = data
        self.slices = slices

    @property
    def raw_file_names(self):
        return [os.path.basename(self.datafile)]

    @property
    def processed_file_names(self):
        name = f'{os.path.basename(self.datafile)[:-4]}_{self.trunc}_{self.seed}'
        if self.save_smiles:
            name += '_smiles'
        if self.read_coord:
            name += '_xtbopt'
        return [f'{name}.pt']
        #return [f'{os.path.basename(self.datafile)[:-4]}_{self.trunc}_{self.seed}.pt' if not self.save_smiles else f'{os.path.basename(self.datafile)[:-4]}_{self.trunc}_{self.seed}_smiles.pt']

    def process(self):
        raw_dataset = np.load(self.datafile, allow_pickle=True)
        shuffle_idx = list(range(len(raw_dataset)))
        if self.seed is not None:
            print(f"[INFO] Shuffling dataset with seed: {self.seed}")
            logging.info(f"[INFO] Shuffling dataset with seed: {self.seed}")
            np.random.seed(self.seed)
            np.random.shuffle(shuffle_idx)
        else:
            print(f"[INFO] Not shuffling dataset")
            logging.info(f"[INFO] Not shuffling dataset")
        raw_dataset = raw_dataset[shuffle_idx]
        
        if self.trunc != 0:
            raw_dataset = raw_dataset[:self.trunc]
        worker_args = [(i, raw_dataset[i], self.seed, self.save_smiles, self.read_coord) for i in range(len(raw_dataset))]
        if self.num_workers > 1:
            logging.info(f"Starting parallel processing with {self.num_workers} cores...")
            

            data_list = []
            
            with Pool(processes=self.num_workers) as pool:

                for result in tqdm(pool.imap(process_single_mol, worker_args), total=len(worker_args)):
                    if result is not None:
                        data_list.append(result)
                    
        else:
            logging.info(f"Starting sequential processing...")
            data_list = []
            for args in tqdm(worker_args):
                result = process_single_mol(args)
                if result is not None:
                    data_list.append(result)
                else:
                    print(f'[WARNING] {args[1]["smiles"]} failed to process, skip')
            

        if len(data_list) == 0:
            raise ValueError("No valid data points were processed.")

        data, slices = self.collate(data_list)
        logging.info(f'[INFO] {len(data_list)} data is saving...')
        torch.save((data, slices), self.processed_paths[0])

    def download(self):
        pass

    def len(self):
        return len(self.slices['x']) - 1

    def get(self, idx):
        return separate(cls=self.data.__class__, batch=self.data, idx=idx, slice_dict=self.slices, decrement=False)
    
def build_graph_input(smiles_lst,batch_size=64):
    data_lst = []
    for smiles in smiles_lst:
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        mol = AllChem.AddHs(mol)


        atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        mult = calculate_gaussview_spin_multiplicity(atom_symbols)
        atom_feat, edge_index, edge_attr, atom_mass = mol2graphinfo(mol, 0, mult)
        data_ = Data(
                x=atom_feat,
                edge_index=edge_index,
                edge_attr=edge_attr,
            )
        data_lst.append(data_)
    dataloader = DataLoader(data_lst,batch_size=batch_size)
    return dataloader