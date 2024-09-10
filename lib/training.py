
from elliot.run import run_experiment
import os
import pandas as pd
import numpy as np
import yaml
import copy
import json





def train(dataset):
    if dataset == 'fairbook' or dataset == 'ml1m':
        train_real(dataset)
    elif dataset == 'ml1m':
        train_ml1m()
    elif dataset == 'epinion':
        train_epinion()
    elif dataset == 'synthetic':
        train_synthetic()
    else:
        print("Error! This is not one of the available datasets.")




def train_real(data):
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





def train_epinion():
    print('Currently unavailable.')



def train_synthetic():
    
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