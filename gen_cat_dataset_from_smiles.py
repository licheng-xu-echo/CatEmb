from catemb.data import CatDataset
#dataset_old = CatDataset(root="../dataset/processed",name="lig_cat_dataset",seed=42,trunc=5000,save_smiles=True)
dataset = CatDataset(root="../dataset/processed",name="lig_cat_dataset_new",seed=42,trunc=0,save_smiles=True)
#dataset_no_smiles = CatDataset(root="../dataset/processed",name="lig_cat_dataset",seed=42,trunc=0,save_smiles=False)