import ipywidgets as widgets
from IPython.display import display
from elliot.run import run_experiment
from lib import tuning, training, processing_recommendations, analysis_recommendations
import os
import pandas as pd
import numpy as np
import yaml
import copy
from scipy import io


def tuning_widget():
    # Define the list of dataset options
    dataset_options = ['fairbook', 'ml1m', 'epinion', 'synthetic']
    
    # Create a radio button widget
    rad_button = widgets.RadioButtons(
        options=dataset_options,
        description='Choose dataset:',
        disabled=False,
        value=None
    )
    
    # Create an OK button widget
    ok_button = widgets.Button(
        description='OK',
        disabled=True
    )
    
    output_radb = widgets.Output()
    
    # Variable to store the selected dataset
    selected_dataset = None
    
    # Define a function to handle the selection change in the radio buttons
    def on_radio_change(change):
        global selected_dataset
        if change['type'] == 'change' and change['name'] == 'value':
            selected_dataset = change['new']
            ok_button.disabled = False  # Enable OK button when a dataset is selected
    
    # Define a function to handle the OK button click
    def on_ok_button_clicked(b):
        global selected_dataset
        with output_radb:
            output_radb.clear_output()  # Clear previous output
            if selected_dataset:
                print(f'Dataset chosen: {selected_dataset}')
                # Trigger your tuning process here
                tuning.tune(selected_dataset)
    
    # Attach the function to the radio button widget
    rad_button.observe(on_radio_change)
    
    # Attach the function to the OK button click
    ok_button.on_click(on_ok_button_clicked)
    
    # Display the radio button widget and the OK button
    display(rad_button, ok_button, output_radb)


def training_widget():
    # Define the list of dataset options
    dataset_options = ['fairbook', 'ml1m', 'epinion', 'synthetic']
    
    # Create a radio button widget
    rad_button = widgets.RadioButtons(
        options=dataset_options,
        description='Choose dataset:',
        disabled=False,
        value=None
    )
    
    # Create an OK button widget
    ok_button = widgets.Button(
        description='OK',
        disabled=True
    )
    
    output_radb = widgets.Output()
    
    # Variable to store the selected dataset
    selected_dataset = None
    
    # Define a function to handle the selection change in the radio buttons
    def on_radio_change(change):
        global selected_dataset
        if change['type'] == 'change' and change['name'] == 'value':
            selected_dataset = change['new']
            ok_button.disabled = False  # Enable OK button when a dataset is selected
    
    # Define a function to handle the OK button click
    def on_ok_button_clicked(b):
        global selected_dataset
        with output_radb:
            output_radb.clear_output()  # Clear previous output
            if selected_dataset:
                print(f'Dataset chosen: {selected_dataset}')
                # Trigger your tuning process here
                training.train(selected_dataset)
    
    # Attach the function to the radio button widget
    rad_button.observe(on_radio_change)
    
    # Attach the function to the OK button click
    ok_button.on_click(on_ok_button_clicked)
    
    # Display the radio button widget and the OK button
    display(rad_button, ok_button, output_radb)


def processing_widget():
    # Define the list of dataset options
    dataset_options = ['fairbook', 'ml1m', 'epinion', 'synthetic']
    
    # Create a radio button widget
    rad_button = widgets.RadioButtons(
        options=dataset_options,
        description='Choose dataset:',
        disabled=False,
        value=None
    )
    
    # Create an OK button widget
    ok_button = widgets.Button(
        description='OK',
        disabled=True
    )
    
    output_radb = widgets.Output()
    
    # Variable to store the selected dataset
    selected_dataset = None
    
    # Define a function to handle the selection change in the radio buttons
    def on_radio_change(change):
        global selected_dataset
        if change['type'] == 'change' and change['name'] == 'value':
            selected_dataset = change['new']
            ok_button.disabled = False  # Enable OK button when a dataset is selected
    
    # Define a function to handle the OK button click
    def on_ok_button_clicked(b):
        global selected_dataset
        with output_radb:
            output_radb.clear_output()  # Clear previous output
            if selected_dataset:
                print(f'Dataset chosen: {selected_dataset}')
                # Trigger your tuning process here
                processing_recommendations.process(selected_dataset)
    
    # Attach the function to the radio button widget
    rad_button.observe(on_radio_change)
    
    # Attach the function to the OK button click
    ok_button.on_click(on_ok_button_clicked)
    
    # Display the radio button widget and the OK button
    display(rad_button, ok_button, output_radb)


def analysis_widget():
    # Define the list of dataset options
    dataset_options = ['fairbook', 'ml1m', 'epinion', 'synthetic']
    
    # Create a radio button widget
    rad_button = widgets.RadioButtons(
        options=dataset_options,
        description='Choose dataset:',
        disabled=False,
        value=None
    )
    
    # Create an OK button widget
    ok_button = widgets.Button(
        description='OK',
        disabled=True
    )
    
    output_radb = widgets.Output()
    
    # Variable to store the selected dataset
    selected_dataset = None
    
    # Define a function to handle the selection change in the radio buttons
    def on_radio_change(change):
        global selected_dataset
        if change['type'] == 'change' and change['name'] == 'value':
            selected_dataset = change['new']
            ok_button.disabled = False  # Enable OK button when a dataset is selected
    
    # Define a function to handle the OK button click
    def on_ok_button_clicked(b):
        global selected_dataset
        with output_radb:
            output_radb.clear_output()  # Clear previous output
            if selected_dataset:
                print(f'Dataset chosen: {selected_dataset}')
                # Trigger your tuning process here
                analysis_recommendations.analyse(selected_dataset)
    
    # Attach the function to the radio button widget
    rad_button.observe(on_radio_change)
    
    # Attach the function to the OK button click
    ok_button.on_click(on_ok_button_clicked)
    
    # Display the radio button widget and the OK button
    display(rad_button, ok_button, output_radb)
