import numpy as np
import torch,os
from .model import CL2D3DMol
from torch_scatter import scatter
from .data import build_graph_input
cur_path = os.path.dirname(__file__)
class CatEmb():
    def __init__(self, model_path=f"{cur_path}/model_path/dim64LN", device='cpu'):
        self.model_path = model_path
        self.device = device
        
        self.full_params = np.load(f"{self.model_path}/full_params.npy",allow_pickle=True).item()
        self.model_params = torch.load(f"{self.model_path}/best_model.pt", map_location=torch.device('cpu'))
        
        
        clmodel = CL2D3DMol(param_2d=self.full_params['param_2d'], param_3d=self.full_params['param_3d'],reduce=self.full_params['param_cl']['reduce'])
        clmodel.load_state_dict(self.model_params['model_state_dict'])
        clmodel.to(self.device)
        clmodel.eval()
        self.clmodel = clmodel
    def gen_desc(self, smiles_lst, batch_size=64):
        dataloader = build_graph_input(smiles_lst,batch_size)
        cat_emb_merge = []
        for data in dataloader:
            data = data.to(self.device)
            atom_feat_2d = self.clmodel.graph_encoder_2d(data.x,data.edge_index,data.edge_attr)
            atom_feat_from2d = self.clmodel.projection_from_2d(atom_feat_2d)
            mol_feat_from_2d = scatter(atom_feat_from2d, data.batch, dim=0, reduce=self.clmodel.reduce)
            cat_emb = mol_feat_from_2d.cpu().detach().numpy()
            cat_emb_merge.append(cat_emb)
        cat_emb_merge = np.concatenate(cat_emb_merge,axis=0)
        return cat_emb_merge