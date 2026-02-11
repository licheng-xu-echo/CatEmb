import torch,math,logging
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import radius_graph
from .data import NUM_BOND_TYPE,NUM_BOND_DIRECTION,NUM_BOND_STEREO,NUM_BOND_INRING,NUM_BOND_ISCONJ,NUM_ATOM_TYPE,NUM_DEGRESS_TYPE,NUM_FORMCHRG_TYPE,\
                           NUM_HYBRIDTYPE,NUM_CHIRAL_TYPE,NUM_AROMATIC_NUM,NUM_VALENCE_TYPE,NUM_Hs_TYPE,NUM_RS_TPYE,NUM_RADICAL_TYPES
from .dependencies.edge_rot_mat import init_edge_rot_mat
from .dependencies.gaussian_rbf import GaussianRadialBasisLayer
from .dependencies.input_block import EdgeDegreeEmbedding
from .dependencies.layer_norm import (
    EquivariantLayerNormArray,
    EquivariantLayerNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonicsV2,
    get_normalization_layer,
)
from .dependencies.module_list import ModuleListInfo
from .dependencies.radial_function import RadialFunction
from .dependencies.scn.smearing import GaussianSmearing
from .dependencies.so3 import CoefficientMappingModule, SO3_Embedding, SO3_Grid, SO3_LinearV2, SO3_Rotation
from .dependencies.transformer_block import FeedForwardNetwork, SO2EquivariantGraphAttention, TransBlockV2

from torch_scatter import scatter_add
import torch.nn.functional as F
from torch_scatter import scatter
from .dependencies.digutils import swish,init,update_e,update_u,update_v,emb,xyz_to_dat
_AVG_NUM_NODES = 23.45625289498063
_AVG_DEGREE = 16.515172958374023

class GCNConv(MessagePassing):
    # adapted from https://github.com/licheng-xu-echo/RXNGraphormer
    def __init__(self, emb_dim, aggr = "add", bond_feat_red="mean"):
        super().__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(NUM_BOND_TYPE, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(NUM_BOND_DIRECTION, emb_dim)
        self.edge_embedding3 = torch.nn.Embedding(NUM_BOND_STEREO, emb_dim)
        self.edge_embedding4 = torch.nn.Embedding(NUM_BOND_INRING, emb_dim)
        self.edge_embedding5 = torch.nn.Embedding(NUM_BOND_ISCONJ, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding3.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding4.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding5.weight.data)
        
        self.edge_embedding_lst = [self.edge_embedding1, self.edge_embedding2, self.edge_embedding3, self.edge_embedding4, self.edge_embedding5]

        self.aggr = aggr
        self.bond_feat_red = bond_feat_red

    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    
    def forward(self, x, edge_index, edge_attr):
        edge_index,_ = add_self_loops(edge_index, num_nodes = x.size(0))
        self_loop_attr = torch.zeros(x.size(0), len(self.edge_embedding_lst))
        self_loop_attr[:,0] = NUM_BOND_TYPE - 1  #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)
        edge_embeddings = []
        for i in range(edge_attr.shape[1]):
            edge_embeddings.append(self.edge_embedding_lst[i](edge_attr[:,i]))
        if self.bond_feat_red == "mean":
            edge_embeddings = torch.stack(edge_embeddings).mean(dim=0)
        elif self.bond_feat_red == "sum":
            edge_embeddings = torch.stack(edge_embeddings).sum(dim=0)
        else:
            raise ValueError("Invalid bond feature reduction method. Please choose from 'mean' or 'sum'")
        
        norm = self.norm(edge_index, x.size(0), x.dtype)

        x = self.linear(x)
        return self.propagate(edge_index=edge_index, aggr=self.aggr, x=x, edge_attr=edge_embeddings, norm = norm)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)
    
class GraphEncoder(nn.Module):
    
    def __init__(self, gnum_layer=4, emb_dim=256, gnn_aggr="add", bond_feat_red="mean", gnn_type='gcn', JK="last", drop_ratio=0.0, node_readout="sum", **kwargs):
        super().__init__()
        self.gnum_layer = gnum_layer
        self.emb_dim = emb_dim
        self.gnn_aggr = gnn_aggr
        self.gnn_type = gnn_type
        self.JK = JK
        self.drop_ratio = drop_ratio
        self.node_readout = node_readout
        assert self.gnum_layer >= 2, "Number of GraphEncoder layers must be greater than 1."

        self.x_embedding1 = torch.nn.Embedding(NUM_ATOM_TYPE, self.emb_dim)     ## atom type
        self.x_embedding2 = torch.nn.Embedding(NUM_DEGRESS_TYPE, self.emb_dim)  ## atom degree
        self.x_embedding3 = torch.nn.Embedding(NUM_FORMCHRG_TYPE, self.emb_dim) ## formal charge
        self.x_embedding4 = torch.nn.Embedding(NUM_HYBRIDTYPE, self.emb_dim)    ## hybrid type
        self.x_embedding5 = torch.nn.Embedding(NUM_CHIRAL_TYPE, self.emb_dim)   ## chiral type
        self.x_embedding6 = torch.nn.Embedding(NUM_AROMATIC_NUM, self.emb_dim)  ## aromatic or not
        self.x_embedding7 = torch.nn.Embedding(NUM_VALENCE_TYPE, self.emb_dim)  ## valence
        self.x_embedding8 = torch.nn.Embedding(NUM_Hs_TYPE, self.emb_dim)       ## number of Hs
        self.x_embedding9 = torch.nn.Embedding(NUM_RS_TPYE, self.emb_dim)       ## R or S
        self.x_embedding10 = torch.nn.Embedding(NUM_RADICAL_TYPES, self.emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding3.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding4.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding5.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding6.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding7.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding8.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding9.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding10.weight.data)
        
        self.x_emedding_lst = [self.x_embedding1,self.x_embedding2,self.x_embedding3,
                               self.x_embedding4,self.x_embedding5,self.x_embedding6,
                               self.x_embedding7,self.x_embedding8,self.x_embedding9,
                               self.x_embedding10]

        ## List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(self.gnum_layer):
            if self.gnn_type.lower() == 'gcn':
                self.gnns.append(GCNConv(self.emb_dim,aggr=self.gnn_aggr,bond_feat_red=bond_feat_red))
            else:
                raise ValueError(f"Unknown GNN type: {self.gnn_type.lower()}")
                
        ## List of layernorms
        self.layer_norms = torch.nn.ModuleList()
        for layer in range(self.gnum_layer):
            self.layer_norms.append(torch.nn.LayerNorm(self.emb_dim))
    
    def forward(self, x, edge_index, edge_attr):
        x_emb_lst = []
        for i in range(x.shape[1]):
            _x_emb = self.x_emedding_lst[i](x[:,i])
            x_emb_lst.append(_x_emb)
        if self.node_readout == 'sum':
            x_emb = torch.stack(x_emb_lst).sum(dim=0)
        elif self.node_readout == 'mean':
            x_emb = torch.stack(x_emb_lst).mean(dim=0)
        h_list = [x_emb]
        for layer in range(self.gnum_layer):
            h = self.gnns[layer](h_list[layer],edge_index=edge_index,edge_attr=edge_attr)
            h = self.layer_norms[layer](h)
            if layer == self.gnum_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=True)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=True)
            h_list.append(h)
        if self.JK == 'last':
            node_representation = h_list[-1]
        elif self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "max":
            h_list = [h.unsqueeze(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0)
        elif self.JK == "sum":
            h_list = [h.unsqueeze(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)
        elif self.JK == "mean":
            h_list = [h.unsqueeze(0) for h in h_list]
            node_representation = torch.mean(torch.cat(h_list, dim = 0), dim = 0)
        elif self.JK == 'last+first':
            node_representation = h_list[-1] + h_list[0]
        else:
            raise NotImplementedError
        
        return node_representation
    
class EquiformerV2(nn.Module):
    """
    Adapted from Equiformer with graph attention built upon SO(2) convolution and feedforward network built upon S2 activation

    Args:
        use_pbc (bool):         Use periodic boundary conditions
        regress_forces (bool):  Compute forces
        otf_graph (bool):       Compute graph On The Fly (OTF)
        max_neighbors (int):    Maximum number of neighbors per atom
        max_radius (float):     Maximum distance between nieghboring atoms in Angstroms
        max_num_elements (int): Maximum atomic number

        num_layers (int):             Number of layers in the GNN
        sphere_channels (int):        Number of spherical channels (one set per resolution)
        attn_hidden_channels (int): Number of hidden channels used during SO(2) graph attention
        num_heads (int):            Number of attention heads
        attn_alpha_head (int):      Number of channels for alpha vector in each attention head
        attn_value_head (int):      Number of channels for value vector in each attention head
        ffn_hidden_channels (int):  Number of hidden channels used during feedforward network
        norm_type (str):            Type of normalization layer (['layer_norm', 'layer_norm_sh', 'rms_norm_sh'])

        lmax_list (int):              List of maximum degree of the spherical harmonics (1 to 10)
        mmax_list (int):              List of maximum order of the spherical harmonics (0 to lmax)
        grid_resolution (int):        Resolution of SO3_Grid

        num_sphere_samples (int):     Number of samples used to approximate the integration of the sphere in the output blocks

        edge_channels (int):                Number of channels for the edge invariant features
        use_atom_edge_embedding (bool):     Whether to use atomic embedding along with relative distance for edge scalar features
        share_atom_edge_embedding (bool):   Whether to share `atom_edge_embedding` across all blocks
        use_m_share_rad (bool):             Whether all m components within a type-L vector of one channel share radial function weights
        distance_function ("gaussian", "sigmoid", "linearsigmoid", "silu"):  Basis function used for distances

        attn_activation (str):      Type of activation function for SO(2) graph attention
        use_s2_act_attn (bool):     Whether to use attention after S2 activation. Otherwise, use the same attention as Equiformer
        use_attn_renorm (bool):     Whether to re-normalize attention weights
        ffn_activation (str):       Type of activation function for feedforward network
        use_gate_act (bool):        If `True`, use gate activation. Otherwise, use S2 activation
        use_grid_mlp (bool):        If `True`, use projecting to grids and performing MLPs for FFNs.
        use_sep_s2_act (bool):      If `True`, use separable S2 activation when `use_gate_act` is False.

        alpha_drop (float):         Dropout rate for attention weights
        drop_path_rate (float):     Drop path rate
        proj_drop (float):          Dropout rate for outputs of attention and FFN in Transformer blocks

        weight_init (str):          ['normal', 'uniform'] initialization of weights of linear layers except those in radial functions
    """

    def __init__(
        self,
        use_pbc=False,
        regress_forces=False,
        otf_graph=True,
        max_neighbors=500,
        max_radius=5.0,
        num_layers=12,
        sphere_channels=128,
        attn_hidden_channels=128,
        num_heads=8,
        attn_alpha_channels=32,
        attn_value_channels=16,
        ffn_hidden_channels=512,
        norm_type="rms_norm_sh",
        lmax_list=[6],
        mmax_list=[2],
        grid_resolution=None,
        num_sphere_samples=128,
        edge_channels=128,
        use_atom_edge_embedding=True,
        share_atom_edge_embedding=False,
        use_m_share_rad=False,
        distance_function="gaussian",
        num_distance_basis=512,
        attn_activation="scaled_silu",
        use_s2_act_attn=False,
        use_attn_renorm=True,
        ffn_activation="scaled_silu",
        use_gate_act=False,
        use_grid_mlp=False,
        use_sep_s2_act=True,
        alpha_drop=0.1,
        drop_path_rate=0.05,
        proj_drop=0.0,
        weight_init="normal",
        energy_rescale_factor=_AVG_NUM_NODES,  # qq
        degree_rescale_factor=_AVG_DEGREE,  # qq
        final='first', # xlc
        **kwargs
    ):
        super().__init__()

        self.use_pbc = use_pbc
        self.regress_forces = regress_forces
        self.otf_graph = otf_graph
        self.max_neighbors = max_neighbors
        self.max_radius = max_radius
        self.cutoff = max_radius
        self.max_num_elements = NUM_ATOM_TYPE
        self.num_layers = num_layers
        self.sphere_channels = sphere_channels
        self.attn_hidden_channels = attn_hidden_channels
        self.num_heads = num_heads
        self.attn_alpha_channels = attn_alpha_channels
        self.attn_value_channels = attn_value_channels
        self.ffn_hidden_channels = ffn_hidden_channels
        self.norm_type = norm_type

        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.grid_resolution = grid_resolution

        self.num_sphere_samples = num_sphere_samples

        self.edge_channels = edge_channels
        self.use_atom_edge_embedding = use_atom_edge_embedding
        self.share_atom_edge_embedding = share_atom_edge_embedding
        if self.share_atom_edge_embedding:
            assert self.use_atom_edge_embedding
            self.block_use_atom_edge_embedding = False
        else:
            self.block_use_atom_edge_embedding = self.use_atom_edge_embedding
        self.use_m_share_rad = use_m_share_rad
        self.distance_function = distance_function
        self.num_distance_basis = num_distance_basis

        self.attn_activation = attn_activation
        self.use_s2_act_attn = use_s2_act_attn
        self.use_attn_renorm = use_attn_renorm
        self.ffn_activation = ffn_activation
        self.use_gate_act = use_gate_act
        self.use_grid_mlp = use_grid_mlp
        self.use_sep_s2_act = use_sep_s2_act

        self.alpha_drop = alpha_drop
        self.drop_path_rate = drop_path_rate
        self.proj_drop = proj_drop

        self.weight_init = weight_init
        assert self.weight_init in ["normal", "uniform"]
        self.final = final
        assert self.final in ['first','mean']

        self.energy_rescale_factor = energy_rescale_factor
        self.degree_rescale_factor = degree_rescale_factor

        self.device = "cpu"  # torch.cuda.current_device()

        self.grad_forces = False
        self.num_resolutions = len(self.lmax_list)
        self.sphere_channels_all = self.num_resolutions * self.sphere_channels

        # Weights for message initialization
        self.sphere_embedding = nn.Embedding(self.max_num_elements, self.sphere_channels_all)

        # Initialize the function used to measure the distances between atoms
        assert self.distance_function in [
            "gaussian",
        ]
        if self.distance_function == "gaussian":
            self.distance_expansion = GaussianSmearing(
                0.0,
                self.cutoff,
                600,
                2.0,
            )
            # self.distance_expansion = GaussianRadialBasisLayer(num_basis=self.num_distance_basis, cutoff=self.max_radius)
        else:
            raise ValueError

        # Initialize the sizes of radial functions (input channels and 2 hidden channels)
        self.edge_channels_list = [int(self.distance_expansion.num_output)] + [self.edge_channels] * 2

        # Initialize atom edge embedding
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            self.source_embedding = nn.Embedding(self.max_num_elements, self.edge_channels_list[-1])
            self.target_embedding = nn.Embedding(self.max_num_elements, self.edge_channels_list[-1])
            self.edge_channels_list[0] = self.edge_channels_list[0] + 2 * self.edge_channels_list[-1]
        else:
            self.source_embedding, self.target_embedding = None, None

        # Initialize the module that compute WignerD matrices and other values for spherical harmonic calculations
        self.SO3_rotation = nn.ModuleList()
        for i in range(self.num_resolutions):
            self.SO3_rotation.append(SO3_Rotation(self.lmax_list[i]))

        # Initialize conversion between degree l and order m layouts
        self.mappingReduced = CoefficientMappingModule(self.lmax_list, self.mmax_list)

        # Initialize the transformations between spherical and grid representations
        self.SO3_grid = ModuleListInfo("({}, {})".format(max(self.lmax_list), max(self.lmax_list)))
        for l in range(max(self.lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(max(self.lmax_list) + 1):
                SO3_m_grid.append(SO3_Grid(l, m, resolution=self.grid_resolution, normalization="component"))
            self.SO3_grid.append(SO3_m_grid)

        # Edge-degree embedding
        self.edge_degree_embedding = EdgeDegreeEmbedding(
            self.sphere_channels,
            self.lmax_list,
            self.mmax_list,
            self.SO3_rotation,
            self.mappingReduced,
            self.max_num_elements,
            self.edge_channels_list,
            self.block_use_atom_edge_embedding,
            rescale_factor=self.degree_rescale_factor,
        )

        # Initialize the blocks for each layer of EquiformerV2
        self.blocks = nn.ModuleList()
        for i in range(self.num_layers):
            block = TransBlockV2(
                self.sphere_channels,
                self.attn_hidden_channels,
                self.num_heads,
                self.attn_alpha_channels,
                self.attn_value_channels,
                self.ffn_hidden_channels,
                self.sphere_channels,
                self.lmax_list,
                self.mmax_list,
                self.SO3_rotation,
                self.mappingReduced,
                self.SO3_grid,
                self.max_num_elements,
                self.edge_channels_list,
                self.block_use_atom_edge_embedding,
                self.use_m_share_rad,
                self.attn_activation,
                self.use_s2_act_attn,
                self.use_attn_renorm,
                self.ffn_activation,
                self.use_gate_act,
                self.use_grid_mlp,
                self.use_sep_s2_act,
                self.norm_type,
                self.alpha_drop,
                self.drop_path_rate,
                self.proj_drop,
            )
            self.blocks.append(block)

        # Output blocks for energy and forces
        self.norm = get_normalization_layer(self.norm_type, lmax=max(self.lmax_list), num_channels=self.sphere_channels)
        self.energy_block = FeedForwardNetwork(
            self.sphere_channels,
            self.ffn_hidden_channels,
            1,
            self.lmax_list,
            self.mmax_list,
            self.SO3_grid,
            self.ffn_activation,
            self.use_gate_act,
            self.use_grid_mlp,
            self.use_sep_s2_act,
        )
        if self.regress_forces:
            self.force_block = SO2EquivariantGraphAttention(
                self.sphere_channels,
                self.attn_hidden_channels,
                self.num_heads,
                self.attn_alpha_channels,
                self.attn_value_channels,
                1,
                self.lmax_list,
                self.mmax_list,
                self.SO3_rotation,
                self.mappingReduced,
                self.SO3_grid,
                self.max_num_elements,
                self.edge_channels_list,
                self.block_use_atom_edge_embedding,
                self.use_m_share_rad,
                self.attn_activation,
                self.use_s2_act_attn,
                self.use_attn_renorm,
                self.use_gate_act,
                self.use_sep_s2_act,
                alpha_drop=0.0,
            )

        self.apply(self._init_weights)
        self.apply(self._uniform_init_rad_func_linear_weights)

    def forward(self, z, pos, batch, edge_index=None):
        batch_size: int = batch.max().item() + 1
        batch_size = int(batch_size)
        self.dtype = pos.dtype
        self.device = pos.device

        atomic_numbers = z.long()
        num_atoms = atomic_numbers.shape[0]

        # qq: simplified, since we do not use pbc
        if self.otf_graph:
            edge_index = radius_graph(
                pos,
                r=self.cutoff,
                batch=batch,
                max_num_neighbors=self.max_neighbors,
            )

        j, i = edge_index[0], edge_index[1]
        edge_distance_vec = pos[j] - pos[i]
        edge_distance = edge_distance_vec.norm(dim=-1)

        ###############################################################
        # Initialize data structures
        ###############################################################

        # Compute 3x3 rotation matrix per edge
        edge_rot_mat = self._init_edge_rot_mat(edge_distance_vec)

        # Initialize the WignerD matrices and other values for spherical harmonic calculations
        for i in range(self.num_resolutions):
            self.SO3_rotation[i].set_wigner(edge_rot_mat)

        ###############################################################
        # Initialize node embeddings
        ###############################################################

        # Init per node representations using an atomic number based embedding
        offset = 0
        x = SO3_Embedding(
            num_atoms,
            self.lmax_list,
            self.sphere_channels,
            self.device,
            self.dtype,
        )

        offset_res = 0
        offset = 0
        # Initialize the l = 0, m = 0 coefficients for each resolution
        for i in range(self.num_resolutions):
            if self.num_resolutions == 1:
                x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers)
            else:
                x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers)[
                    :, offset : offset + self.sphere_channels
                ]
            offset = offset + self.sphere_channels
            offset_res = offset_res + int((self.lmax_list[i] + 1) ** 2)

        # Edge encoding (distance and atom edge)
        edge_distance = self.distance_expansion(edge_distance)
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            source_element = atomic_numbers[edge_index[0]]  # Source atom atomic number
            target_element = atomic_numbers[edge_index[1]]  # Target atom atomic number
            source_embedding = self.source_embedding(source_element)
            target_embedding = self.target_embedding(target_element)
            edge_distance = torch.cat((edge_distance, source_embedding, target_embedding), dim=1)

        # Edge-degree embedding
        edge_degree = self.edge_degree_embedding(atomic_numbers, edge_distance, edge_index)
        x.embedding = x.embedding + edge_degree.embedding

        ###############################################################
        # Update spherical node embeddings
        ###############################################################

        for i in range(self.num_layers):
            x = self.blocks[i](
                x,
                atomic_numbers,
                edge_distance,
                edge_index,
                batch=batch,  # SO3_Embedding  # for GraphDropPath
            )

        # Final layer norm
        x.embedding = self.norm(x.embedding)
        if self.final == 'first':
            x_3d = x.embedding[:,0,:]
        else:
            x_3d = x.embedding.mean(1)

        ###############################################################
        # Energy estimation
        ###############################################################
        node_energy = self.energy_block(x)
        node_energy = node_energy.embedding.narrow(1, 0, 1)
        energy = torch.zeros(batch_size, device=node_energy.device, dtype=node_energy.dtype)
        energy.index_add_(0, batch, node_energy.view(-1))
        energy = energy / self.energy_rescale_factor
        self.energy = energy            
        return x_3d,energy
        

    # Initialize the edge rotation matrics
    def _init_edge_rot_mat(self, edge_distance_vec):
        return init_edge_rot_mat(edge_distance_vec)

    @property
    def num_params(self):
        return sum([p.numel() for p in self.parameters()])

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear) or isinstance(m, SO3_LinearV2):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            if self.weight_init == "normal":
                std = 1 / math.sqrt(m.in_features)
                torch.nn.init.normal_(m.weight, 0, std)

        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    def _uniform_init_rad_func_linear_weights(self, m):
        if isinstance(m, RadialFunction):
            m.apply(self._uniform_init_linear_weights)

    def _uniform_init_linear_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            std = 1 / math.sqrt(m.in_features)
            torch.nn.init.uniform_(m.weight, -std, std)

    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd_list = []
        named_parameters_list = [name for name, _ in self.named_parameters()]
        for module_name, module in self.named_modules():
            if (
                isinstance(module, torch.nn.Linear)
                or isinstance(module, SO3_LinearV2)
                or isinstance(module, torch.nn.LayerNorm)
                or isinstance(module, EquivariantLayerNormArray)
                or isinstance(module, EquivariantLayerNormArraySphericalHarmonics)
                or isinstance(module, EquivariantRMSNormArraySphericalHarmonics)
                or isinstance(module, EquivariantRMSNormArraySphericalHarmonicsV2)
                or isinstance(module, GaussianRadialBasisLayer)
            ):
                for parameter_name, _ in module.named_parameters():
                    if isinstance(module, torch.nn.Linear) or isinstance(module, SO3_LinearV2):
                        if "weight" in parameter_name:
                            continue
                    global_parameter_name = module_name + "." + parameter_name
                    assert global_parameter_name in named_parameters_list
                    no_wd_list.append(global_parameter_name)
        return set(no_wd_list)
    
class DimeNetPP(nn.Module):
    r"""
        The re-implementation for DimeNet++ from the `"Fast and Uncertainty-Aware Directional Message Passing for Non-Equilibrium Molecules" <https://arxiv.org/abs/2011.14115>`_ paper
        under the 3DGN gramework from `"Spherical Message Passing for 3D Molecular Graphs" <https://openreview.net/forum?id=givsRXsOt9r>`_ paper.
        
        Args:
            energy_and_force (bool, optional): If set to :obj:`True`, will predict energy and take the negative of the derivative of the energy with respect to the atomic positions as predicted forces. (default: :obj:`False`)
            cutoff (float, optional): Cutoff distance for interatomic interactions. (default: :obj:`5.0`)
            num_layers (int, optional): Number of building blocks. (default: :obj:`4`)
            hidden_channels (int, optional): Hidden embedding size. (default: :obj:`128`)
            out_channels (int, optional): Size of each output sample. (default: :obj:`1`)
            int_emb_size (int, optional): Embedding size used for interaction triplets. (default: :obj:`64`)
            basis_emb_size (int, optional): Embedding size used in the basis transformation. (default: :obj:`8`)
            out_emb_channels (int, optional): Embedding size used for atoms in the output block. (default: :obj:`256`)
            num_spherical (int, optional): Number of spherical harmonics. (default: :obj:`7`)
            num_radial (int, optional): Number of radial basis functions. (default: :obj:`6`)
            envelop_exponent (int, optional): Shape of the smooth cutoff. (default: :obj:`5`)
            num_before_skip (int, optional): Number of residual layers in the interaction blocks before the skip connection. (default: :obj:`1`)
            num_after_skip (int, optional): Number of residual layers in the interaction blocks before the skip connection. (default: :obj:`2`)
            num_output_layers (int, optional): Number of linear layers for the output blocks. (default: :obj:`3`)
            act: (function, optional): The activation funtion. (default: :obj:`swish`) 
            output_init: (str, optional): The initialization fot the output. It could be :obj:`GlorotOrthogonal` and :obj:`zeros`. (default: :obj:`GlorotOrthogonal`)       
    """
    def __init__(
        self, energy_and_force=False, cutoff=5.0, num_layers=4, 
        hidden_channels=128, out_channels=128, int_emb_size=64, basis_emb_size=8, out_emb_channels=256, 
        num_spherical=7, num_radial=6, envelope_exponent=5, 
        num_before_skip=1, num_after_skip=2, num_output_layers=3, 
        act=swish, output_init='GlorotOrthogonal',**kwargs):
        super().__init__()

        self.cutoff = cutoff
        self.energy_and_force = energy_and_force

        self.init_e = init(num_radial, hidden_channels, act)
        self.init_v = update_v(hidden_channels, out_emb_channels, out_channels, num_output_layers, act, output_init)
        self.init_u = update_u()
        self.emb = emb(num_spherical, num_radial, self.cutoff, envelope_exponent)
        
        self.update_vs = nn.ModuleList([
            update_v(hidden_channels, out_emb_channels, out_channels, num_output_layers, act, output_init) for _ in range(num_layers)])

        self.update_es = nn.ModuleList([
            update_e(
                hidden_channels, int_emb_size, basis_emb_size,
                num_spherical, num_radial,
                num_before_skip, num_after_skip,
                act,
            )
            for _ in range(num_layers)
        ])

        self.update_us = torch.nn.ModuleList([update_u() for _ in range(num_layers)])

        self.reset_parameters()

    def reset_parameters(self):
        self.init_e.reset_parameters()
        self.init_v.reset_parameters()
        self.emb.reset_parameters()
        for update_e in self.update_es:
            update_e.reset_parameters()
        for update_v in self.update_vs:
            update_v.reset_parameters()


    def forward(self, z, pos, batch, edge_index=None):
        if edge_index is None:
            edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        num_nodes=z.size(0)
        dist, angle, i, j, idx_kj, idx_ji = xyz_to_dat(pos, edge_index, num_nodes, use_torsion=False)

        emb = self.emb(dist, angle, idx_kj)

        #Initialize edge, node, graph features
        e = self.init_e(z, emb, i, j)
        v = self.init_v(e, i)
        u = self.init_u(torch.zeros_like(scatter(v, batch, dim=0)), v, batch) #scatter(v, batch, dim=0)
        
        for update_e, update_v, update_u in zip(self.update_es, self.update_vs, self.update_us):
            e = update_e(e, emb, idx_kj, idx_ji)
            v = update_v(e, i)
            u = update_u(u, v, batch) #u += scatter(v, batch, dim=0)
        self.e = e
        self.v = v
        self.u = u
        u_sum = u.sum(1)
        return v,u_sum

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.0, ln=True):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.ln = ln
        self.fc = torch.nn.ModuleList()
        assert self.num_layers >= 2

        self.fc.append(nn.Linear(input_dim, hidden_dim))
        for layer in range(num_layers - 1):
            #self.fc.append(nn.Dropout(p=dropout))
            self.fc.append(nn.ReLU())
            if self.ln:
                self.fc.append(nn.LayerNorm(self.hidden_dim))
            if layer < num_layers - 2:
                self.fc.append(nn.Linear(hidden_dim, hidden_dim))
        self.fc.append(nn.Linear(hidden_dim, output_dim))
        self.fc = nn.Sequential(*self.fc)

    def forward(self, x):
        return self.fc(x)
    
class CoordEncoder(nn.Module):
    def __init__(self, model_type='equif', param_3d={
        'use_pbc':False,
        'regress_forces':False,
        'otf_graph':True,
        'max_neighbors':64,
        'max_radius':5.0,
        'num_layers':6,
        'sphere_channels':128,
        'attn_hidden_channels':128,
        'num_heads':4,
        'attn_alpha_channels':32,
        'attn_value_channels':16,
        'ffn_hidden_channels':64,
        'norm_type':"rms_norm_sh",
        'lmax_list':[4],
        'mmax_list':[2],
        'grid_resolution':None,
        'num_sphere_samples':128,
        'edge_channels':128,
        'use_atom_edge_embedding':True,
        'share_atom_edge_embedding':False,
        'use_m_share_rad':False,
        'distance_function':"gaussian",
        'num_distance_basis':64,
        'attn_activation':"scaled_silu",
        'use_s2_act_attn':False,
        'use_attn_renorm':True,
        'ffn_activation':"scaled_silu",
        'use_gate_act':False,
        'use_grid_mlp':False,
        'use_sep_s2_act':True,
        'alpha_drop':0.1,
        'drop_path_rate':0.05,
        'proj_drop':0.0,
        'weight_init':"normal",
        'final':'mean' # xlc
}):
        super().__init__()
        if model_type == 'equif':
            
            self.encoder = EquiformerV2(**param_3d)
        elif model_type == 'dimenetpp':
            self.encoder = DimeNetPP(**param_3d)
        else:
            raise NotImplementedError
        self.model_type = model_type
        self.param_3d = param_3d

    def forward(self, z, pos, batch, edge_index=None):
        emb_3d,energy_p = self.encoder(z, pos, batch, edge_index)
        return emb_3d,energy_p