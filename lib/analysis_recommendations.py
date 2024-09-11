import pickle as pkl
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from tqdm import tqdm
from itertools import chain

import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf


def highest_average(df_list, column_name = 'recommendation'):
    highest_average = -(10**6)
    highest_i = -1
    for i in range(len(df_list)):
        df = df_list[i]
        mean = np.mean(df[column_name].values)
        print('mean', mean)
        if mean > highest_average:
            highest_average = mean
            highest_i = i
    # print(highest_average, highest_i)
    return highest_average, highest_i
    
def mannwhitneyu_test(df_list, alt = 'greater', column_name = 'recommendation'):
    # find the highest average 
    
    ha, hi = highest_average(df_list, column_name)
    print('Highest version: ', hi, ha)
    inds_df_list = list(range(len(df_list)))
    to_test_inds = inds_df_list[:hi] + inds_df_list[hi+1:]
    # print(to_test_inds)
    df1 = df_list[hi]
    pvalues = []
    for ind in to_test_inds:
        df2 = df_list[ind]
        x = df1[column_name].values
        y = df2[column_name].values
        print(np.mean(x), np.mean(y))
        pvalue = mannwhitneyu(x,y, alternative = alt)[1]
        pvalues.append(pvalue)
    return [(to_test_inds[i],pvalues[i]) for i in range(len(pvalues))] # pvalues for all comparisons


def analyse(dataset):
    if dataset == 'fairbook' or dataset == 'ml1m':
        analyse_real(dataset)
    elif dataset == 'epinion':
        print('Currently unavailable.')
    elif dataset == 'synthetic':
        analyse_synthetic()
    else:
        print("Error! This is not one of the available datasets.")


def analyse_real(data_strategy):

    
    mlp_values = ['64-32', '64-64'] # the different versions of the algorithm tested
    algo_name = "DMF"
    file_location = "metrics/" + algo_name + "/" +data_strategy+"/"
    results = []
    for mlp in mlp_values:
        file = open(file_location + data_strategy + "_" + mlp + ".pkl", "rb")
        result = pkl.load(file)
        results.append(result)
    index = pd.MultiIndex.from_product(
        [mlp_values],
        names=["Network layers"],
    ).drop_duplicates()
    results = pd.DataFrame(results, index=index)
    detailed_results = []
    for mlp in mlp_values:
        file = open(file_location + 'detailed_per_item_'+data_strategy + "_" + mlp + ".pkl", "rb")
        result = pkl.load(file)
        detailed_results.append(result)

    
    metrics_order = ["pop_corr", "ARP", "ave_PL", "ACLT", "AggDiv", "RMSE", "NDCG"]
    metrics = results[metrics_order]
    metrics = metrics.rename(
        columns={"pop_corr": "PopCorr", "ave_PL": "PL", "ACLT": "APLT", "NDCG": "NDCG@10"}
    )
    metrics['RealPopCorr'] = metrics.PopCorr.apply(lambda x: x[0])
    metrics['Significance'] = metrics.PopCorr.apply(lambda x: True if x[1]<0.005 else False)
    metrics['PopCorr'] = metrics.RealPopCorr 
    metrics = metrics.drop('RealPopCorr', axis=1)

    with open("metrics/"+algo_name+'/'+data_strategy+'/'+data_strategy+"_final_metrics.pkl", "wb") as f:
        pkl.dump(metrics.round(3).drop('RMSE',axis=1), f)  # RMSE is irrelevant for DMF

    print(metrics.round(3).drop('RMSE', axis=1).to_latex())

    # # Significance tests
    print(mlp_values) 
    print('Use the above to figure out significance comparisons.')
    
    # ## 1. Average Recommendation Popularity

    print("ARP:")
    print(mannwhitneyu_test(detailed_results))
    
    
    # ## 2. Popularity Lift
    
    for df in detailed_results:
        df['popularity_lift'] = (df['recommendation']-df['profile'])/df['profile']*100
    print("PL:")
    print(mannwhitneyu_test(detailed_results, column_name = 'popularity_lift')) 
    






def analyse_synthetic():

    mlp_values = ['64-32', '64-64'] # the different versions of the algorithm tested
    algo_name = "DMF"
    file_location = "metrics/" + algo_name + "/"

    data_strategies = [
        "uniformly_random",
        "popularity_good",
        "popularity_bad",
        "popularity_good_for_bp_ur",
        # "popularity_bad_for_bp_ur",
    ]
    
    all_results = []
    for data_strategy in data_strategies:
        results = []
        for mlp in mlp_values:
            file = open(file_location + data_strategy+'/'+data_strategy + "_" + mlp + ".pkl", "rb")
            result = pkl.load(file)
            results.append(result)
        all_results.append(results)
    ds = ["Scenario 1", "Scenario 2", "Scenario 3", "Scenario 4"
          # , "Scenario 5"
     ]
    index = pd.MultiIndex.from_product(
        [ds, mlp_values], names=["DataStrategy", "Factors"]
    ).drop_duplicates()

    unlisted_sl = list(chain(*all_results))
    relevant_values = ["pop_corr", "ARP", "ave_PL", "NDCG"]
    final_results = pd.DataFrame(unlisted_sl, index=index)[relevant_values]

    dict_detailed={}
    for data_strategy in data_strategies:
        detailed_results = []
        for mlp in mlp_values:
            file = open(file_location +data_strategy+ '/detailed_per_item_'+data_strategy + "_" + mlp + ".pkl", "rb")
            result = pkl.load(file)
            detailed_results.append(result)
        dict_detailed[data_strategy] = detailed_results

    metrics_order = ["pop_corr", "ARP", "ave_PL", "NDCG"]
    metrics = final_results[metrics_order]
    metrics = metrics.rename(
        columns={"pop_corr": "PopCorr", "ave_PL": "PL",  "NDCG": "NDCG@10"}
    )
    metrics['RealPopCorr'] = metrics.PopCorr.apply(lambda x: x[0])
    metrics['Significance'] = metrics.PopCorr.apply(lambda x: True if x[1]<0.005 else False)
    metrics['PopCorr'] = metrics.RealPopCorr 
    metrics = metrics.drop('RealPopCorr', axis=1)

    with open("metrics/"+algo_name+'/synthetic_final_metrics.pkl', "wb") as f:
        pkl.dump(metrics.round(3), f) 
    print(metrics.round(3).drop('Significance',axis=1).to_latex())

    for data_strategy in data_strategies:
        print(data_strategy)
        results = dict_detailed[data_strategy]
        # # Significance tests
        print(mlp_values) 
        print('Use the above to figure out significance comparisons.')
        
        
        # ## 1. Average Recommendation Popularity
    
        print("ARP:")
        print(mannwhitneyu_test(results))
        
        
        # ## 2. Popularity Lift
        
        for df in results:
            df['popularity_lift'] = (df['recommendation']-df['profile'])/df['profile']*100
        print("PL:")
        print(mannwhitneyu_test(results, column_name = 'popularity_lift')) 
        print("--------------------------------------------------------------")