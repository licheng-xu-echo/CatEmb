import numpy as np
from copy import deepcopy

def random_init(cat_score_lst, leave_best_topk=3, batch_size=5, rand_seed=42):
    print(f"best {leave_best_topk} catalyst label:",[item[1] for item in cat_score_lst[-leave_best_topk:]])
    if leave_best_topk > 0:
        target_cat = cat_score_lst[-leave_best_topk:]     ## Top-k
        remain_cat = cat_score_lst[:-leave_best_topk]     ## 43 - k
    else:
        target_cat = []
        remain_cat = cat_score_lst
    shuffle_idx = np.arange(len(remain_cat))
    np.random.seed(rand_seed)
    np.random.shuffle(shuffle_idx)
    start_idx_lst = shuffle_idx[:batch_size]                 ## 5 from 40
    pool_idx_lst = shuffle_idx[batch_size:]                ## 35
    start_cat = [remain_cat[idx] for idx in start_idx_lst]
    pool_cat = [remain_cat[idx] for idx in pool_idx_lst] + target_cat
    start_score = np.array([cat[1] for cat in start_cat])


    ######
    cur_cat = deepcopy(start_cat)
    cur_score = deepcopy(start_score)
    if len(cur_score) > 0:
        max_score = np.max(cur_score)
    else:
        max_score = 0
    print(f"{len(pool_cat)}, {len(cur_cat)}")
    print(f"Max score: {max_score:.4f}, {len(cur_score)}")
    
    return cur_score,cur_cat,pool_cat


def recommend_by_dist(catemb_calc,cur_score,cur_cat,pool_cat,batch_size=5,best_n=1):
    ######
    cur_cat_smi_lst = [cur[0] for cur in cur_cat]
    pool_cat_smi_lst = [pool[0] for pool in pool_cat]
    
    best_topk_cat_idx_lst = np.abs(cur_score).argsort()[::-1][:best_n]     # best one from existed catalysts
                        
    cur_cat_desc_arr = catemb_calc.gen_desc(cur_cat_smi_lst)
    pool_cat_desc_arr = catemb_calc.gen_desc(pool_cat_smi_lst)


    best_existed_cat_desc_arr = cur_cat_desc_arr[best_topk_cat_idx_lst]
    dist_of_pool_to_best = np.array([np.linalg.norm(arr - pool_cat_desc_arr,axis=1) for arr in best_existed_cat_desc_arr])


    recommend_cat_idx_lst_by_dist = dist_of_pool_to_best.sum(0).argsort()[:batch_size]
    recommend_cat = [pool_cat[idx] for idx in recommend_cat_idx_lst_by_dist]
    recommend_score = np.array([cat[1] for cat in recommend_cat])

    cur_cat = cur_cat + recommend_cat
    cur_cat_smi_lst = [cat[0] for cat in cur_cat]
    cur_score = np.concatenate([cur_score,recommend_score])
    pool_cat = [cat for idx,cat in enumerate(pool_cat) if not cat[0] in cur_cat_smi_lst]
    pool_cat_smi_lst = [cat[0] for cat in pool_cat]
    print(f"{len(pool_cat)}, {len(cur_cat)}")
    print(f"Max score: {np.max(cur_score):.4f}, {len(cur_score)}")
    return cur_score,cur_cat,pool_cat

def recommend_by_random(cur_score,cur_cat,pool_cat,batch_size=5):
    ######
    chose_idx_lst = np.random.choice(len(pool_cat),batch_size,replace=False)
    recommend_cat = [pool_cat[idx] for idx in chose_idx_lst]
    recommend_score = np.array([cat[1] for cat in recommend_cat])

    cur_cat = cur_cat + recommend_cat
    cur_cat_smi_lst = [cat[0] for cat in cur_cat]
    cur_score = np.concatenate([cur_score,recommend_score])
    pool_cat = [cat for idx,cat in enumerate(pool_cat) if not cat[0] in cur_cat_smi_lst]
    print(f"{len(pool_cat)}, {len(cur_cat)}")
    print(f"Max score: {np.max(cur_score):.4f}, {len(cur_score)}")
    return cur_score,cur_cat,pool_cat

def recommend_by_model(cur_score,cur_cat,pool_cat,rct12_pdt_pair_set,rxn_smi_fp_map,rxn_smi_label_map,model,batch_size=5):
    
    cur_cat_smi_lst = [item[0] for item in cur_cat]
    pool_cat_smi_lst = [item[0] for item in pool_cat]
    cur_rxn_smi_lst = []
    pool_rxn_smi_lst = []
    for cat_smi in cur_cat_smi_lst:
        for rct12_pdt_pair in rct12_pdt_pair_set:
            rxn_smi = f"{rct12_pdt_pair[0]}.{rct12_pdt_pair[1]}>{cat_smi}>{rct12_pdt_pair[2]}"
            cur_rxn_smi_lst.append(rxn_smi)
    for cat_smi in pool_cat_smi_lst:
        for rct12_pdt_pair in rct12_pdt_pair_set:
            rxn_smi = f"{rct12_pdt_pair[0]}.{rct12_pdt_pair[1]}>{cat_smi}>{rct12_pdt_pair[2]}"
            pool_rxn_smi_lst.append(rxn_smi)
    cur_train_x = np.array([rxn_smi_fp_map[smi] for smi in cur_rxn_smi_lst])
    cur_train_y = np.array([rxn_smi_label_map[smi] for smi in cur_rxn_smi_lst])
    pool_x = np.array([rxn_smi_fp_map[smi] for smi in pool_rxn_smi_lst])
    model.fit(cur_train_x,cur_train_y)
    pool_pred = model.predict(pool_x)
    pool_cat_smi_pred_map = {}
    for rxn_smi,pred in zip(pool_rxn_smi_lst,pool_pred):
        cat_smi = rxn_smi.split(">")[1]
        if cat_smi not in pool_cat_smi_pred_map:
            pool_cat_smi_pred_map[cat_smi] = []
        pool_cat_smi_pred_map[cat_smi].append(pred)
    pool_cat_smi_ave_pred_lst = sorted([[cat_smi,np.mean(pred_lst)] for cat_smi,pred_lst in pool_cat_smi_pred_map.items()],key=lambda x:-x[1])
    recommend_cat_smi_lst = [item[0] for item in pool_cat_smi_ave_pred_lst[:batch_size]]
    recommend_cat = [cat for cat in pool_cat if cat[0] in recommend_cat_smi_lst]
    recommend_score = np.array([cat[1] for cat in recommend_cat])

    cur_cat = cur_cat + recommend_cat
    cur_cat_smi_lst = [cat[0] for cat in cur_cat]
    cur_score = np.concatenate([cur_score,recommend_score])
    pool_cat = [cat for idx,cat in enumerate(pool_cat) if not cat[0] in cur_cat_smi_lst]
    pool_cat_smi_lst = [cat[0] for cat in pool_cat]
    print(f"{len(pool_cat)}, {len(cur_cat)}")
    print(f"Max score: {np.max(cur_score):.4f}, {len(cur_score)}")
    return cur_score,cur_cat,pool_cat