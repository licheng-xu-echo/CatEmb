import numpy as np
import logging,argparse,torch
from rdkit import Chem,RDLogger
from rdkit.Chem import AllChem
from torch_geometric.data import InMemoryDataset,Data
from torch_geometric.data.separate import separate
from catemb.data import mol2graphinfo,CatDataset,get_idx_split
from catemb.utils import calculate_gaussview_spin_multiplicity,symbol_pos_to_xyz_file,setup_logger,str_to_bool
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from datetime import datetime
from catemb.train import CLTrain
from catemb.model import CL2D3DMol
from torch_geometric.data import DataLoader
RDLogger.DisableLog('rdApp.*')
pt = Chem.GetPeriodicTable()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="../dataset/processed", help="data path")
    parser.add_argument("--data_file", type=str, default="lig_cat_dataset", help="data file")
    parser.add_argument("--trunc", type=int, default=0, help="truncation of dataset")
    parser.add_argument("--read_coord", type=str_to_bool, default=False, help="read coordinates from data file")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--gnum_layer_2d", type=int, default=4, help="number of graph convolutional layers in 2d model")
    parser.add_argument("--gnum_layer_3d", type=int, default=4, help="number of graph convolutional layers in 3d model")
    parser.add_argument("--emb_dim_2d", type=int, default=128, help="embedding dimension in 2d model")
    parser.add_argument("--model_type_3d", type=str, default="equif", help="model type for 3d model")
    parser.add_argument("--emb_dim_3d", type=int, default=128, help="embedding dimension in 3d model")
    parser.add_argument("--emb_dest_dim", type=int, default=64, help="embedding destination dimension")
    parser.add_argument("--ln", type=str_to_bool, default=True, help="use layer normalization")
    parser.add_argument("--cl_weight", type=float, default=1.0, help="weight of contrastive loss")
    parser.add_argument("--kl_weight", type=float, default=1.0, help="weight of kl loss")
    parser.add_argument("--e_weight", type=float, default=1.0, help="weight of energy loss")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--scheduler_type", type=str, default="reduceonplateau", help="scheduler type", choices=["reduceonplateau","steplr","noamlr"])
    parser.add_argument("--lr_decay_factor", type=float, default=0.95, help="learning rate decay factor")
    parser.add_argument("--lr_decay_step_size", type=int, default=10, help="learning rate decay step size")
    parser.add_argument("--warmup_step", type=int, default=50000, help="warmup step")
    parser.add_argument("--temperature", type=float, default=0.1, help="temperature")
    parser.add_argument("--loss_metric", type=str, default="InfoNCE_dot_prod", help="loss metric", choices=["InfoNCE_dot_prod","EBM_dot_prod"])
    parser.add_argument("--epoch", type=int, default=20, help="number of epochs")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--tag", type=str, default="test", help="tag for model")
    parser.add_argument("--save_path", type=str, default="./save_model", help="save path for model")
    parser.add_argument("--device", type=str, default="cuda:0", help="device to use")
    
    args = parser.parse_args()
    param_data = {"root":args.root,"name":args.data_file,"trunc":args.trunc,'train_ratio':0.9,'valid_ratio':0.1,'batch_size':args.batch_size,'save_smiles':True,'read_coord':args.read_coord}
    param_2d = {'gnum_layer':args.gnum_layer_2d, 'emb_dim':args.emb_dim_2d, 'dest_dim':args.emb_dest_dim, 'gnn_aggr':"add", 'bond_feat_red':"mean", 
                'gnn_type':'gcn', 'JK':"last", 'drop_ratio':0.0, 'node_readout':"sum", 'ln':args.ln}

    param_3d_equif = {
            'model_type':'equif',
            'use_pbc':False,
            'regress_forces':False,
            'otf_graph':True,
            'max_neighbors':64,
            'max_radius':5.0,
            'num_layers':args.gnum_layer_3d,
            'sphere_channels':args.emb_dim_3d,
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
            'final':'first', # xlc
            'ln':args.ln
    }

    param_3d_dimenetpp = {
            'model_type':'dimenetpp',
            'cutoff':5.0, 'num_layers':args.gnum_layer_3d, 
            'hidden_channels':128, 'out_channels':args.emb_dim_3d, 'int_emb_size':64, 'basis_emb_size':8, 'out_emb_channels':256, 
            'num_spherical':7, 'num_radial':6, 'envelope_exponent':5, 
            'num_before_skip':1, 'num_after_skip':2, 'num_output_layers':3, 
            'output_init':'GlorotOrthogonal',
            'ln':args.ln
    }

    if args.model_type_3d == "equif":
        param_3d = param_3d_equif
    elif args.model_type_3d == "dimenetpp":
        param_3d = param_3d_dimenetpp
    else:
        raise ValueError("model_type_3d must be either 'equif' or 'dimenetpp'")

    param_cl = {"metric":args.loss_metric,
                "T":args.temperature,
                "cl_weight":args.cl_weight,
                "kl_weight":args.kl_weight,
                "e_weight":args.e_weight,
                "reduce":"mean",
                }
    param_optimizer = {'type':'adamw','lr':args.lr}
    param_scheduler = {'type': args.scheduler_type, 'lr_decay_step_size': args.lr_decay_step_size, 'lr_decay_factor': args.lr_decay_factor, 'min_lr': 5e-6, 'warmup_step':args.warmup_step}
    param_other = {"log_iter_step":10, 'clip_norm':50.0, 'epoch':args.epoch,'save_path':args.save_path,'device':args.device,"seed":args.seed, "tag":args.tag}
    if not torch.cuda.is_available():
        param_other['device'] = "cpu"
    dt = datetime.strftime(datetime.now(), "%y%m%d-%H%Mh")
    param_other['save_path'] = f"{param_other['save_path']}/{dt}_{args.tag}"

    full_params = {"param_data":param_data,
                "param_2d":param_2d,
                "param_3d":param_3d,
                "param_cl":param_cl,
                "param_optimizer":param_optimizer,
                "param_scheduler":param_scheduler,
                "param_other":param_other}

    setup_logger(param_other['save_path'])
    np.save(f"{param_other['save_path']}/full_params.npy",full_params)
    logging.info(str(args))
    logging.info(str(full_params))
    new_dataset = CatDataset(**param_data,seed=param_other['seed'])
    data_split_dict = get_idx_split(len(new_dataset),
                                    int(param_data['train_ratio']*len(new_dataset)),
                                    int(param_data['valid_ratio']*len(new_dataset)),seed=param_other['seed'])
    train_dataset = new_dataset[data_split_dict['train']]
    valid_dataset = new_dataset[data_split_dict['valid']]

    train_dataloader = DataLoader(train_dataset, batch_size=param_data['batch_size'], shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=param_data['batch_size'], shuffle=False)
    clmodel = CL2D3DMol(param_2d,param_3d,reduce=param_cl['reduce'])
    cltrain = CLTrain(clmodel,param_3d,param_cl,param_optimizer,param_scheduler,param_other)

    cltrain.run(train_dataloader,valid_dataloader)


if __name__ == "__main__":
    main()