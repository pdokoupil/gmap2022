from pathlib import Path
import sys, os
from typing import Dict, List, NamedTuple, Tuple
import numpy as np
from collections import defaultdict
import time

import argparse

from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer

EPSILON = 1e-6

# fold, directory
def get_folds(data_dir: str) -> List[Tuple[int, str]]:
    folds = []
    for dir in [f for f in Path(data_dir).iterdir() if f.is_dir()]:
        dir_name = os.path.basename(dir)
        if str(dir_name).isnumeric():
            folds.append((int(dir_name), str(dir)))
    folds.sort()
    return folds

# returns 2d numpy array where 1. index is userId and 2. index is itemId, values are float ratings
def load_data(data_dir: str, fold: int) -> np.ndarray:
    return np.load(os.path.join(data_dir, str(fold), "mf_data.npy"))

class Group(NamedTuple):
    id: int
    members: List[int]

# group data must be in file formated with groupId, userid1, userid2...
# separated by tabs
def load_group_data(data_dir: str, group_type: str, group_size: int) -> List[Group]:
    groups = []
    filename = group_type + "_group_" + str(group_size)
    path = os.path.join(data_dir, filename)
    with open(path) as group_file:
        lines = group_file.readlines()
        for line in lines:
            items = line.replace('\n', '').split("\t")
            items = list(map(int, items))
            groups.append(Group(items[0], items[1:]))
            if len(items) < group_size + 1:
                raise Exception("Group file invalid: " + path)
    return groups
    
    
    
def get_recommendation_files(data_dir: str, fold: int, group: str, group_size: int) -> List[str]:
    rec_path = os.path.join(data_dir, str(fold), group, str(group_size)) 
    return list([str(f) for f in Path(rec_path).iterdir() if f.is_file()])

class AlgRecommendations(NamedTuple):
    alg_name: str
    # dict indexed by groupId
    group_recommendations: Dict[int, List[int]] = {} 


# items are sorted from best to worst
# returns list of tuples where first is the agreg name and second is dictionary of recommendations indexed by group id
def load_agregated_recommendations(data_dir: str, fold: int, group: str, group_size: int) -> List[AlgRecommendations]:
    files = get_recommendation_files(data_dir, fold, group, group_size)
    returnList = []
    for file in files:
        recommendationsMap = defaultdict(list) 
        with open(file) as recommendation_file:
            lines = recommendation_file.readlines()
            for line in lines:
                items = line.replace('\n', '').split("\t")[:2]
                items = list(map(int, items))
                group_id = items[0]
                recommendationsMap[group_id].append(items[1])
        alg_name = os.path.basename(file)
        returnList.append(AlgRecommendations(alg_name, recommendationsMap))
    return returnList

#calculates discounted cumulative gain on the array of relevances
def calculate_dcg(values):
    values = np.array(values)
    if values.size: #safety check
        return np.sum(values / np.log2(np.arange(2, values.size + 2)))
    return 0.0    

#order items of user, cut best topk_size, calculate DCG of the cut
#test_data = uidxoid matrix of ratings
#topk_size = volume of items per user on which to calculate IDCG
#return dictionary {userID:IDCG_value}
def calculate_per_user_IDCG(test_data, topk_size):
    users = range(test_data.shape[0])
    idcg_per_user = {}
    for user in users:        
        per_user_items = test_data[user] 
        sorted_items = np.sort(per_user_items)[::-1]
        sorted_items = sorted_items[0:20]
        
        idcg = calculate_dcg(sorted_items)
        idcg_per_user[user] = idcg
        
        #print(sorted_items)
        #print(idcg)
        #exit()
        
    return idcg_per_user
        
    

class Result(NamedTuple):
    alg: str
    metric: str
    value: float


##### Non-user rating normalizations #####
def norm_shift_nonlinear(rating_matrix, **kwargs):
    return np.maximum(EPSILON, rating_matrix + kwargs["normalization_c"])
      

##### User-level rating normalizations #####
def u_norm_min_max_scaler(rating_matrix, **kwargs):
    scaler = MinMaxScaler()
    return scaler.fit_transform(rating_matrix.T).T


def compute_metrics(test_data: np.ndarray, groups: List[Group], alg_data: AlgRecommendations, args) -> List[Result]:
    # test_data are triplets: user_id, item_id, and rating
    #LP: test data is matrix user_id x item_id !!!!!! a ja si rikal, jakto ze ti to prirazeni funguje...
    idcg_per_user = calculate_per_user_IDCG(test_data, 20)
    
    
    avg_rating = []
    min_rating = []
    minmax_rating = []
    std_rating = []
    
    avg_nDCG_rating = []
    min_nDCG_rating = []
    minmax_nDCG_rating = []
    std_nDCG_rating = []
        
    for group in groups:
        group_users_sum_ratings = []
        group_users_ndcg_ratings = []
        group_id = group.id 
        rec_for_group = alg_data.group_recommendations[group_id]
        for group_user_id in group.members:
            user_sum = 0.0
            user_list = []
            for item_id in rec_for_group:
                rating = test_data[group_user_id, item_id]
                #print(group_user_id, item_id, rating)
                #print(type(test_data))
                #print(test_data.shape)
                #print(test_data[group_user_id])
                #exit()
                user_sum += rating
                user_list.append(rating)
            ndcg = calculate_dcg(user_list) / idcg_per_user[group_user_id]   
            
            group_users_sum_ratings.append(user_sum)
            group_users_ndcg_ratings.append(ndcg)
        #TODO: quick&dirty code - consider revising   
        
        group_users_mean_ratings = [i/len(rec_for_group) for i in group_users_sum_ratings] 
        avg_rating.append(np.average(group_users_mean_ratings)) 
        min = np.min(group_users_mean_ratings)
        min_rating.append(min) 
        max = np.max(group_users_mean_ratings)
        minmax_rating.append(0.0 if max == 0.0 else min/max)
        std_rating.append(np.std(group_users_mean_ratings)) 
        
        avg_nDCG_rating.append(np.average(group_users_ndcg_ratings)) 
        min = np.min(group_users_ndcg_ratings)
        min_nDCG_rating.append(min) 
        max = np.max(group_users_ndcg_ratings)
        minmax_nDCG_rating.append(0.0 if max == 0.0 else min/max)
        std_nDCG_rating.append(np.std(group_users_ndcg_ratings))         
        
    results = []
    results.append(Result(alg_data.alg_name, "AR_avg", np.average(avg_rating)))
    results.append(Result(alg_data.alg_name, "AR_min", np.average(min_rating)))
    results.append(Result(alg_data.alg_name, "AR_min/max", np.average(minmax_rating)))
    results.append(Result(alg_data.alg_name, "AR_std", np.average(std_rating)))
    
    results.append(Result(alg_data.alg_name, "nDCG_avg", np.average(avg_nDCG_rating)))
    results.append(Result(alg_data.alg_name, "nDCG_min", np.average(min_nDCG_rating)))
    results.append(Result(alg_data.alg_name, "nDCG_min/max", np.average(minmax_nDCG_rating)))
    results.append(Result(alg_data.alg_name, "nDCG_std", np.average(std_nDCG_rating)))    

    return results


def process_fold(groups: List[Group], data_dir: str, fold: int, group: str, group_size: int, args) -> List[Result]:
    algs_data = load_agregated_recommendations(data_dir, fold, group, group_size)
    test_data = load_data(data_dir, fold)

    # Normalizations
    if args.user_rating_normalization != "identity":
        user_rating_normalization = globals()[args.user_rating_normalization]
        test_data = user_rating_normalization(test_data, **vars(args))
    
    rating_normalization = globals()[args.rating_normalization]
    test_data = rating_normalization(test_data, **vars(args))
    
    if args.use_quadratic_amplification:
        test_data = test_data ** 2


    results = []
    for alg_data in algs_data:
        results.extend(compute_metrics(test_data, groups, alg_data, args))
    #for result in results:
    #    print(result)
    return results

def main(data_folder, group_type, group_size, args):
    print(data_folder, group_type, group_size)
    folds = get_folds(data_folder)
    groups: List[Group] = load_group_data(data_folder, group_type, int(group_size))
    
    results = []
    for fold, _ in folds:
        results.extend(process_fold(groups, data_folder, fold, group_type, int(group_size), args))

        
    algs = set(map(lambda x:x.alg, results))
    metrics = list(set(map(lambda x:x.metric, results)))
    print(metrics)
    metrics.sort()
    print(metrics)
    res = "alg,group_type,group_size" + "," + ",".join(metrics)+"\n"
    for alg in algs:
        values = [alg, group_type, str(group_size)]
        for metric in metrics:
            value = np.average([v.value for v in results if v.alg == alg and v.metric == metric])
            value = round(value,3)
            values.append(str(value))
        res += ",".join(values)+"\n"
    return res


parser = argparse.ArgumentParser()
parser.add_argument("--rating_normalization", type=str)
parser.add_argument("--user_rating_normalization", type=str)
parser.add_argument("--use_quadratic_amplification", action="store_true", default=False)
parser.add_argument("--normalization_c", type=float, help="Constant used rating normalization")
parser.add_argument("--use_all_constants", action="store_true", default=False)
parser.add_argument("--path_prefix", type=str, default=".")
parser.add_argument("--group_types", type=str)
parser.add_argument("--group_sizes", type=str)
args = parser.parse_args()

all_constants_with_u_norm = np.linspace(-0.6,0.6,7)
all_constants_without_u_norm = np.linspace(-3.0,3.0,7)

all_group_types = ["sim"]
all_group_sizes = ["2", "4", "8"]

if __name__ == "__main__":

    if not args.user_rating_normalization:
        args.user_rating_normalization = "identity"

    if not args.group_types:
        print(f"Group types not specified, using: {all_group_types}")
        group_types = all_group_types
    else:
        group_types = args.group_types.split(",")

    if not args.group_sizes:
        print(f"Group size not specified, using: {all_group_sizes}")
        group_sizes = all_group_sizes
    else:
        group_sizes = args.group_sizes.split(",")

    print(f"Group types: {group_types}, group sizes: {group_sizes}")

    for group_type in group_types: #["sim", "div", "random"]:
        for group_size in group_sizes: #["2","3","4","8"]:
            if args.use_all_constants:

                if args.user_rating_normalization == "identity":
                    all_constants = all_constants_without_u_norm
                else:
                    all_constants = all_constants_with_u_norm

                print(f"Used constants are: {all_constants}")

                for c in all_constants:
                    #f = open("results/result_"+group_type+"_"+group_size+"_c="+c,"w")
                    args.normalization_c = c
                    out_file_name = f"results/result_{group_type}_{group_size}_c={c}_{args.rating_normalization}_{args.user_rating_normalization}_{args.use_quadratic_amplification}"
                    f = open(os.path.join(args.path_prefix, out_file_name),"w")
                    results = main(os.path.join(args.path_prefix, "data/ml1m"), group_type, group_size, args)
                    f.write(results)
            else:
                #f = open("results/result_"+group_type+"_"+group_size+"_c="+args.normalization_c,"w")
                out_file_name = f"results/result_{group_type}_{group_size}_c={args.normalization_c}_{args.rating_normalization}_{args.user_rating_normalization}_{args.use_quadratic_amplification}"
                f = open(os.path.join(args.path_prefix, out_file_name),"w")
                results = main(os.path.join(args.path_prefix, "data/ml1m"), group_type, group_size, args)
                f.write(results)

    #args = sys.argv[1:]
    #print(args)
    #main(args[0], args[1], args[2])
    #main("data/ml1m", "sim", "2")