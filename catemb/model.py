from .encoder import GraphEncoder,CoordEncoder,MLP
from torch import nn
from torch_scatter import scatter
class CL2D3DMol(nn.Module):
    def __init__(self,param_2d,param_3d,reduce='mean'):
        super().__init__()
        self.param_2d = param_2d
        self.param_3d = param_3d
        self.reduce = reduce
        self.graph_encoder_2d = GraphEncoder(**param_2d)
        self.graph_encoder_3d = CoordEncoder(model_type=param_3d['model_type'],param_3d=param_3d)
        self.projection_from_2d = MLP(input_dim=param_2d['emb_dim'],
                              hidden_dim=param_2d['emb_dim'],
                              output_dim=param_2d['dest_dim'],
                              ln=param_2d['ln'])
        if param_3d['model_type'] == 'equif':
            self.projection_from_3d = MLP(input_dim=param_3d['sphere_channels'],
                                        hidden_dim=param_3d['sphere_channels'],
                                        output_dim=param_2d['dest_dim'],
                                        ln=param_3d['ln'])
        elif param_3d['model_type'] == 'dimenetpp':
            self.projection_from_3d = MLP(input_dim=param_3d['out_channels'],
                                        hidden_dim=param_3d['out_channels'],
                                        output_dim=param_2d['dest_dim'],
                                        ln=param_3d['ln'])
            
        else:
            raise ValueError(f'model type {param_3d["model_type"]} not supported')
    
    def forward(self, data):
        atom_feat_2d = self.graph_encoder_2d(data.x,data.edge_index,data.edge_attr)
        atom_feat_3d,energy_p = self.graph_encoder_3d(data.x[:,0],data.mol_coords,data.batch,edge_index=data.edge_index)
        atom_feat_from2d = self.projection_from_2d(atom_feat_2d)
        atom_feat_from3d = self.projection_from_3d(atom_feat_3d)
        mol_feat_from_2d = scatter(atom_feat_from2d, data.batch, dim=0, reduce=self.reduce)
        mol_feat_from_3d = scatter(atom_feat_from3d, data.batch, dim=0, reduce=self.reduce)
        return mol_feat_from_2d, mol_feat_from_3d, energy_p