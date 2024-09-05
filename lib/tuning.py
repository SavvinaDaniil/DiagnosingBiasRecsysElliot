#!/usr/bin/env python
# coding: utf-8

# In[15]:


from elliot.run import run_experiment
from scipy import io
import os
import pandas as pd
import numpy as np
import yaml
import copy


# In[ ]:
def tune(dataset):
    if dataset == 'fairbook':
        tune_fairbook()
    elif dataset == 'ml1m':
        tune_ml1m()
    elif dataset == 'epinion':
        tune_epinion()
    elif dataset == 'synthetic':
        tune_synthetic()
    else:
        print("Error! This is not one of the available datasets.")


def tune_fairbook():
    # possible values for the hyperparameters
    mlp_values = ['(64,32)'
                  , '(64,64)'
                 ]
    with open('config_files/fairbook_tune.yml', 'r') as f: # open the relevant yaml file
        base_config = yaml.safe_load(f)
    
    for mlp in mlp_values:
        print("We re doing the following: ", mlp)
        # Make a copy of the base configuration
        config = copy.deepcopy(base_config)
        # Update the configuration with the current hyperparameters
        config['experiment']['models']['DMF']['user_mlp'] = mlp
        config['experiment']['models']['DMF']['item_mlp'] = mlp
    
        # Write the configuration to a temporary file
        with open('config_files/temp_config.yml', 'w') as f:
            yaml.dump(config, f)
    
        # Run the experiment with the current configuration
        run_experiment('config_files/temp_config.yml')
        
        
        # Remove the temp file
        os.remove('config_files/temp_config.yml')    


# In[ ]:


def tune_ml1m():
    mlp_values = ['(64,32)'
                  , '(64,64)'
                 ]
    ratings = pd.read_csv(
    'data/movielens_1m/ml1m_events.dat', header=None, sep="::", engine="python"
).drop(3, axis=1)
    ratings.columns = ["user", "item", "rating"]
    np.savetxt("data/movielens_1m/ml1m.tsv", ratings, delimiter='\t',fmt='%i')
    with open('config_files/ml1m_tune.yml', 'r') as f: # open the relevant yaml file
        base_config = yaml.safe_load(f)
    
    for mlp in mlp_values:
        print("We re doing the following: ", mlp)
        # Make a copy of the base configuration
        config = copy.deepcopy(base_config)
        # Update the configuration with the current hyperparameters
        config['experiment']['models']['DMF']['user_mlp'] = mlp
        config['experiment']['models']['DMF']['item_mlp'] = mlp
    
        # Write the configuration to a temporary file
        with open('config_files/temp_config.yml', 'w') as f:
            yaml.dump(config, f)
    
        # Run the experiment with the current configuration
        run_experiment('config_files/temp_config.yml')
        
        
        # Remove the temp file
        os.remove('config_files/temp_config.yml')    


# In[ ]:


def tune_epinion():
    mlp_values = ['(64,32)'
                  , '(64,64)'
                 ]
    data = "epinion"
    mat = io.loadmat('data/epinion/epinion_events.mat')
    mat_df = pd.DataFrame(mat["rating_with_timestamp"])
    mat_df.columns = ["user", "item", ".", "rating", "..", "..."]
    ratings = mat_df[["user", "item", "rating"]]
    ratings = ratings.drop_duplicates(subset=["user", "item"], keep="last").reset_index(drop=True)
    np.savetxt("data/epinion/epinion.tsv", ratings, delimiter='\t',fmt='%i')

    with open('config_files/epinion_tune.yml', 'r') as f: # open the relevant yaml file
        base_config = yaml.safe_load(f)
    
    for mlp in mlp_values:
        print("We re doing the following: ", mlp)
        # Make a copy of the base configuration
        config = copy.deepcopy(base_config)
        # Update the configuration with the current hyperparameters
        config['experiment']['models']['DMF']['user_mlp'] = mlp
        config['experiment']['models']['DMF']['item_mlp'] = mlp
    
        # Write the configuration to a temporary file
        with open('config_files/temp_config.yml', 'w') as f:
            yaml.dump(config, f)
    
        # Run the experiment with the current configuration
        run_experiment('config_files/temp_config.yml')
        
        
        # Remove the temp file
        os.remove('config_files/temp_config.yml')    
        


# In[ ]:


def tune_synthetic_data():
    mlp_values = ['(64,32)'
                  , '(64,64)'
                 ]
    data_strategies = [
        "uniformly_random",
        "popularity_good",
        "popularity_bad",
        "popularity_good_for_bp_ur",
        "popularity_bad_for_bp_ur",
    ]
    for data_strategy in data_strategies:
        csv_file = pd.read_csv("data/"+data_strategy+"/"+data_strategy+".csv")
        np.savetxt("data/"+data_strategy+"/"+data_strategy+".tsv", csv_file, delimiter='\t',fmt='%i')
    for data_strategy in data_strategies:
        print(data_strategy)
        with open('config_files/'+data_strategy+'_tune.yml', 'r') as f: # open the relevant yaml file
            base_config = yaml.safe_load(f)
        for mlp in mlp_values:
            print("We re doing the following: ", mlp)
            # Make a copy of the base configuration
            config = copy.deepcopy(base_config)
            # Update the configuration with the current hyperparameters
            config['experiment']['models']['DMF']['user_mlp'] = mlp
            config['experiment']['models']['DMF']['item_mlp'] = mlp
        
            # Write the configuration to a temporary file
            with open('config_files/temp_config.yml', 'w') as f:
                yaml.dump(config, f)
        
            # Run the experiment with the current configuration
            run_experiment('config_files/temp_config.yml')
            
            
            # Remove the temp file
            os.remove('config_files/temp_config.yml')    
        