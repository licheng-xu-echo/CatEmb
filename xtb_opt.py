import os,argparse,time
from subprocess import run,PIPE

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tgt_dir",type=str,default="/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/8359-xulicheng/CatEmb/dataset/processed/xtb_opt_new")
    parser.add_argument("--start_idx",type=int,default=0)
    parser.add_argument("--end_idx",type=int,default=2000)

    args = parser.parse_args()


    tgt_dir = args.tgt_dir
    start_idx = args.start_idx
    end_idx = args.end_idx


    xyz_folders = sorted(os.listdir(tgt_dir),key=lambda x:int(x))
    ct = 0
    start_time = time.time()
    for idx in range(start_idx,end_idx):
        print(f"{idx} start [from {start_idx} to {end_idx}] ({len(xyz_folders)} in total)")
        
        xyz_folder = xyz_folders[idx]
        xyz_folder = os.path.join(tgt_dir,xyz_folder)
        os.chdir(xyz_folder)
        if os.path.exists(f"{xyz_folder}/xtbopt.xyz"):
            print(f"{idx} already done [from {start_idx} to {end_idx}] ({len(xyz_folders)} in total)")
            continue
        with open(f"{idx}.xyz","r") as fr:
            lines = fr.readlines()
        title = lines[1].strip()
        title_blks = title.split()
        chrg = int(title_blks[3])
        uhf = int(title_blks[7])
        cmd = f"xtb {idx}.xyz --opt --chrg {chrg} --uhf {uhf} --input constrain.inp > xtboptlog.out"
        run(cmd,stdout=PIPE,stderr=PIPE,universal_newlines=True,cwd=None,shell=True,executable='/bin/bash',check=False)
        print(f"{idx} done [from {start_idx} to {end_idx}] ({len(xyz_folders)} in total)")
        ct += 1
        if ct % 100 == 0:
            end_time = time.time()
            print(f"{ct} file(s) have done, cost {end_time-start_time:.2f} seconds")
if __name__ == "__main__":
    main()