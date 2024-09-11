import pickle as pkl
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from tqdm import tqdm

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
        # print('mean', mean)
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
    






def process_synthetic():
    
    data = 'uniformly_random'
    
    
    location = 'results/'+data+'/performance/'

    json_files = [best_params for best_params in os.listdir(location) if best_params.endswith('.json')]


    for filename in json_files: # corresponds to mlp
    
        file = location + filename
        with open(file) as f:
            d = json.load(f)
            mlp = d[1]['configuration']['item_mlp']
            best_lr = d[1]['configuration']['lr']
        for i in range(1, 6):
            print('Start for ', i, '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            with open('config_files/'+data+str(i)+'.yml', 'r') as f: # open the relevant yaml file
                base_config = yaml.safe_load(f)
            
            
            # Make a copy of the base configuration
            config = copy.deepcopy(base_config)
            # Update the configuration with the current hyperparameters
            config['experiment']['models']['DMF']['user_mlp'] = mlp
            config['experiment']['models']['DMF']['item_mlp'] = mlp
            config['experiment']['models']['DMF']['lr'] = best_lr
        
            # Write the configuration to a temporary file
            with open('config_files/temp_config.yml', 'w') as f:
                yaml.dump(config, f)
        
            # Run the experiment with the current configuration
            run_experiment('config_files/temp_config.yml')
            
            
            # Remove the temp file
            os.remove('config_files/temp_config.yml')
    
    
    # ## Train Popularity good
    
 
    
    
    data = 'popularity_good'
    location = 'results/'+data+'/performance/'

    json_files = [best_params for best_params in os.listdir(location) if best_params.endswith('.json')]


    for filename in json_files: # corresponds to mlp
    
        file = location + filename
        with open(file) as f:
            d = json.load(f)
            mlp = d[1]['configuration']['item_mlp']
            best_lr = d[1]['configuration']['lr']
        for i in range(1, 6):
            print('Start for ', i, '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            with open('config_files/'+data+str(i)+'.yml', 'r') as f: # open the relevant yaml file
                base_config = yaml.safe_load(f)
            
            
            # Make a copy of the base configuration
            config = copy.deepcopy(base_config)
            # Update the configuration with the current hyperparameters
            config['experiment']['models']['DMF']['user_mlp'] = mlp
            config['experiment']['models']['DMF']['item_mlp'] = mlp
            config['experiment']['models']['DMF']['lr'] = best_lr
        
            # Write the configuration to a temporary file
            with open('config_files/temp_config.yml', 'w') as f:
                yaml.dump(config, f)
        
            # Run the experiment with the current configuration
            run_experiment('config_files/temp_config.yml')
            
            
            # Remove the temp file
            os.remove('config_files/temp_config.yml')  
    
    
    # ## Train Popularity bad
    
    
    
    data = 'popularity_bad'    
    
    location = 'results/'+data+'/performance/'

    json_files = [best_params for best_params in os.listdir(location) if best_params.endswith('.json')]


    for filename in json_files: # corresponds to mlp
    
        file = location + filename
        with open(file) as f:
            d = json.load(f)
            mlp = d[1]['configuration']['item_mlp']
            best_lr = d[1]['configuration']['lr']
        for i in range(1, 6):
            print('Start for ', i, '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            with open('config_files/'+data+str(i)+'.yml', 'r') as f: # open the relevant yaml file
                base_config = yaml.safe_load(f)
            
            
            # Make a copy of the base configuration
            config = copy.deepcopy(base_config)
            # Update the configuration with the current hyperparameters
            config['experiment']['models']['DMF']['user_mlp'] = mlp
            config['experiment']['models']['DMF']['item_mlp'] = mlp
            config['experiment']['models']['DMF']['lr'] = best_lr
        
            # Write the configuration to a temporary file
            with open('config_files/temp_config.yml', 'w') as f:
                yaml.dump(config, f)
        
            # Run the experiment with the current configuration
            run_experiment('config_files/temp_config.yml')
            
            
            # Remove the temp file
            os.remove('config_files/temp_config.yml')    
    
    # ## Train Popularity good big
        
    
    data = 'popularity_good_for_bp_ur'
    location = 'results/'+data+'/performance/'

    json_files = [best_params for best_params in os.listdir(location) if best_params.endswith('.json')]


    for filename in json_files: # corresponds to mlp
    
        file = location + filename
        with open(file) as f:
            d = json.load(f)
            mlp = d[1]['configuration']['item_mlp']
            best_lr = d[1]['configuration']['lr']
        for i in range(1, 6):
            print('Start for ', i, '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            with open('config_files/popularity_good_big'+str(i)+'.yml', 'r') as f: # open the relevant yaml file
                base_config = yaml.safe_load(f)
            
            
            # Make a copy of the base configuration
            config = copy.deepcopy(base_config)
            # Update the configuration with the current hyperparameters
            config['experiment']['models']['DMF']['user_mlp'] = mlp
            config['experiment']['models']['DMF']['item_mlp'] = mlp
            config['experiment']['models']['DMF']['lr'] = best_lr
        
            # Write the configuration to a temporary file
            with open('config_files/temp_config.yml', 'w') as f:
                yaml.dump(config, f)
        
            # Run the experiment with the current configuration
            run_experiment('config_files/temp_config.yml')
            
            
            # Remove the temp file
            os.remove('config_files/temp_config.yml')  
    
    
    # ## Train Popularity bad big
    
    
    
    data = 'popularity_bad_for_bp_ur'
    location = 'results/'+data+'/performance/'

    json_files = [best_params for best_params in os.listdir(location) if best_params.endswith('.json')]


    for filename in json_files: # corresponds to mlp
    
        file = location + filename
        with open(file) as f:
            d = json.load(f)
            mlp = d[1]['configuration']['item_mlp']
            best_lr = d[1]['configuration']['lr']
        for i in range(1, 6):
            print('Start for ', i, '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            with open('config_files/popularity_bad_big'+str(i)+'.yml', 'r') as f: # open the relevant yaml file
                base_config = yaml.safe_load(f)
            
            
            # Make a copy of the base configuration
            config = copy.deepcopy(base_config)
            # Update the configuration with the current hyperparameters
            config['experiment']['models']['DMF']['user_mlp'] = mlp
            config['experiment']['models']['DMF']['item_mlp'] = mlp
            config['experiment']['models']['DMF']['lr'] = best_lr
        
            # Write the configuration to a temporary file
            with open('config_files/temp_config.yml', 'w') as f:
                yaml.dump(config, f)
        
            # Run the experiment with the current configuration
            run_experiment('config_files/temp_config.yml')
            
            
            # Remove the temp file
            os.remove('config_files/temp_config.yml')