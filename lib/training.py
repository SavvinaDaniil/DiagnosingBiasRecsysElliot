
from elliot.run import run_experiment
import os
import pandas as pd
import numpy as np
import yaml
import copy
import json





def train(dataset):
    if dataset == 'fairbook':
        train_fairbook()
    elif dataset == 'ml1m':
        train_ml1m()
    elif dataset == 'epinion':
        train_epinion()
    elif dataset == 'synthetic':
        train_synthetic()
    else:
        print("Error! This is not one of the available datasets.")




def train_fairbook():
    data = 'fairbook'
    location = 'results/'+data+'/performance/'
    filename = 'bestmodelparams_cutoff_10_relthreshold_0_2024_08_01_08_46_55.json'
    file = location + filename
    with open(file) as f:
        d = json.load(f)
        mlp = d[1]['configuration']['item_mlp']
        assert mlp == '(64,64)'
        best_lr = d[1]['configuration']['lr']
    for i in range(1, 6):
        print('Start for ', i, '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        with open('config_files/fairbook'+str(i)+'.yml', 'r') as f: # open the relevant yaml file
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

    filename = 'bestmodelparams_cutoff_10_relthreshold_0_2024_07_31_11_32_14.json'
    file = location + filename
    with open(file) as f:
        d = json.load(f)
        mlp = d[1]['configuration']['item_mlp']
        assert mlp == '(64,32)'
        best_lr = d[1]['configuration']['lr']
    for i in range(1, 6):
        print('Start for ', i, '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        with open('config_files/fairbook'+str(i)+'.yml', 'r') as f: # open the relevant yaml file
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


def train_ml1m():


    data = 'ml1m'
    location = 'results/'+data+'/performance/'
    
    
    # 64 - 64
    
    # Set filename manually!
    filename = 'bestmodelparams_cutoff_10_relthreshold_0_2024_08_28_14_17_42.json'
    
    file = location + filename
    with open(file) as f:
        d = json.load(f)
        mlp = d[1]['configuration']['item_mlp']
        assert mlp == '(64,64)'
        best_lr = d[1]['configuration']['lr']
    
    
    for i in range(1, 6):
        print('Start for ', i, '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        with open('config_files/ml1m'+str(i)+'.yml', 'r') as f: # open the relevant yaml file
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

    
    # 64 - 32
    
    # Set filename manually!
    filename = 'bestmodelparams_cutoff_10_relthreshold_0_2024_08_28_03_24_41.json'
    
    
    
    
    file = location + filename
    with open(file) as f:
        d = json.load(f)
        mlp = d[1]['configuration']['item_mlp']
        assert mlp == '(64,32)'
        best_lr = d[1]['configuration']['lr']
    
    
    
    
    for i in range(1, 6):
        print('Start for ', i, '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        with open('config_files/ml1m'+str(i)+'.yml', 'r') as f: # open the relevant yaml file
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
    
    
    # In[ ]:
    
    
    location = 'results/'+data+'/performance/'
    
    
    # 64 - 64
    
    # Set filename manually!
    
    # In[ ]:
    
    
    filename = 'bestmodelparams_cutoff_10_relthreshold_0_2024_08_04_02_27_21.json'
    
    
    # In[ ]:
    
    
    file = location + filename
    with open(file) as f:
        d = json.load(f)
        mlp = d[1]['configuration']['item_mlp']
        assert mlp == '(64,64)'
        best_lr = d[1]['configuration']['lr']
    
    
    # In[ ]:
    
    
    for i in range(1, 6):
        print('Start for ', i, '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        with open('config_files/uniformly_random'+str(i)+'.yml', 'r') as f: # open the relevant yaml file
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
    
    
    # In[ ]:
    
    
    
    
    
    # 64 - 32
    
    # Set filename manually!
    
    # In[ ]:
    
    
    filename = 'bestmodelparams_cutoff_10_relthreshold_0_2024_08_03_06_56_39.json'
    
    
    # In[ ]:
    
    
    file = location + filename
    with open(file) as f:
        d = json.load(f)
        mlp = d[1]['configuration']['item_mlp']
        assert mlp == '(64,32)'
        best_lr = d[1]['configuration']['lr']
    
    
    # In[ ]:
    
    
    for i in range(1, 6):
        print('Start for ', i, '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        with open('config_files/uniformly_random'+str(i)+'.yml', 'r') as f: # open the relevant yaml file
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
    
    # In[ ]:
    
    
    data = 'popularity_good'
    
    
    # In[ ]:
    
    
    location = 'results/'+data+'/performance/'
    
    
    # 64 - 64
    
    # Set filename manually!
    
    # In[ ]:
    
    
    filename = 'bestmodelparams_cutoff_10_relthreshold_0_2024_08_05_16_39_20.json'
    
    
    # In[ ]:
    
    
    file = location + filename
    with open(file) as f:
        d = json.load(f)
        mlp = d[1]['configuration']['item_mlp']
        assert mlp == '(64,64)'
        best_lr = d[1]['configuration']['lr']
    
    
    # In[ ]:
    
    
    for i in range(1, 6):
        print('Start for ', i, '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        with open('config_files/popularity_good'+str(i)+'.yml', 'r') as f: # open the relevant yaml file
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
    
    
    # In[ ]:
    
    
    
    
    
    # 64 - 32
    
    # Set filename manually!
    
    # In[ ]:
    
    
    filename = 'bestmodelparams_cutoff_10_relthreshold_0_2024_08_04_21_29_55.json'
    
    
    # In[ ]:
    
    
    file = location + filename
    with open(file) as f:
        d = json.load(f)
        mlp = d[1]['configuration']['item_mlp']
        assert mlp == '(64,32)'
        best_lr = d[1]['configuration']['lr']
    
    
    # In[ ]:
    
    
    for i in range(1, 6):
        print('Start for ', i, '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        with open('config_files/popularity_good'+str(i)+'.yml', 'r') as f: # open the relevant yaml file
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
    
    # In[ ]:
    
    
    data = 'popularity_bad'
    
    
    # In[ ]:
    
    
    location = 'results/'+data+'/performance/'
    
    
    # 64 - 64
    
    # Set filename manually!
    
    # In[ ]:
    
    
    filename = 'bestmodelparams_cutoff_10_relthreshold_0_2024_08_07_06_34_11.json'
    
    
    # In[ ]:
    
    
    file = location + filename
    with open(file) as f:
        d = json.load(f)
        mlp = d[1]['configuration']['item_mlp']
        assert mlp == '(64,64)'
        best_lr = d[1]['configuration']['lr']
    
    
    # In[ ]:
    
    
    for i in range(1, 6):
        print('Start for ', i, '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        with open('config_files/popularity_bad'+str(i)+'.yml', 'r') as f: # open the relevant yaml file
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
    
    
    # In[ ]:
    
    
    
    
    
    # 64 - 32
    
    # Set filename manually!
    
    # In[ ]:
    
    
    filename = 'bestmodelparams_cutoff_10_relthreshold_0_2024_08_06_11_31_53.json'
    
    
    # In[ ]:
    
    
    file = location + filename
    with open(file) as f:
        d = json.load(f)
        mlp = d[1]['configuration']['item_mlp']
        assert mlp == '(64,32)'
        best_lr = d[1]['configuration']['lr']
    
    
    # In[ ]:
    
    
    for i in range(1, 6):
        print('Start for ', i, '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        with open('config_files/popularity_bad'+str(i)+'.yml', 'r') as f: # open the relevant yaml file
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
    
    # In[ ]:
    
    
    data = 'popularity_good_for_bp_ur'
    
    
    # In[ ]:
    
    
    location = 'results/'+data+'/performance/'
    
    
    # 64 - 64
    
    # Set filename manually!
    
    # In[ ]:
    
    
    filename = 'bestmodelparams_cutoff_10_relthreshold_0_2024_08_08_15_33_01.json'
    
    
    # In[ ]:
    
    
    file = location + filename
    with open(file) as f:
        d = json.load(f)
        mlp = d[1]['configuration']['item_mlp']
        assert mlp == '(64,64)'
        best_lr = d[1]['configuration']['lr']
    
    
    # In[ ]:
    
    
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
    
    
    # In[ ]:
    
    
    
    
    
    # 64 - 32
    
    # Set filename manually!
    
    # In[ ]:
    
    
    filename = 'bestmodelparams_cutoff_10_relthreshold_0_2024_08_07_23_47_02.json'
    
    
    # In[ ]:
    
    
    file = location + filename
    with open(file) as f:
        d = json.load(f)
        mlp = d[1]['configuration']['item_mlp']
        assert mlp == '(64,32)'
        best_lr = d[1]['configuration']['lr']
    
    
    # In[ ]:
    
    
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
    
    # In[ ]:
    
    
    data = 'popularity_bad_for_bp_ur'
    
    
    # In[ ]:
    
    
    location = 'results/'+data+'/performance/'
    
    
    # 64 - 64
    
    # Set filename manually!
    
    # In[ ]:
    
    
    filename = 'bestmodelparams_cutoff_10_relthreshold_0_2024_08_09_21_07_34.json'
    
    
    # In[ ]:
    
    
    file = location + filename
    with open(file) as f:
        d = json.load(f)
        mlp = d[1]['configuration']['item_mlp']
        assert mlp == '(64,64)'
        best_lr = d[1]['configuration']['lr']
    
    
    # In[ ]:
    
    
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
    
    
    # In[ ]:
    
    
    
    
    
    # 64 - 32
    
    # Set filename manually!
    
    # In[ ]:
    
    
    filename = 'bestmodelparams_cutoff_10_relthreshold_0_2024_08_09_06_37_14.json'
    
    
    # In[ ]:
    
    
    file = location + filename
    with open(file) as f:
        d = json.load(f)
        mlp = d[1]['configuration']['item_mlp']
        assert mlp == '(64,32)'
        best_lr = d[1]['configuration']['lr']
    
    
    # In[ ]:
    
    
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
    
    
    # In[ ]:
    
    
    
    
