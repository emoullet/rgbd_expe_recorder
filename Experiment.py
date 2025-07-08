import csv
import itertools
import os
import random
import subprocess
import tkinter as tk
from tkinter import messagebox, ttk
import pickle

import threading
import cv2
import pandas as pd
import numpy as np

import databases_utils as db
import ExperimentRecorder as erc
# import ExperimentPreProcessor as epp
# import ExperimentReplayer as erp
# import ExperimentAnalyser as ea

from config import SESSION_OPTIONS, MODES
            
def get_row_and_column_index_from_index(index, nb_items_total):
    # get the number of rows and columns, knowing that the number of columns and rows should be as close as possible
    nb_rows = int(np.sqrt(nb_items_total))
    if nb_rows == 0:
        nb_rows = 1
    nb_columns = int(np.ceil(nb_items_total / nb_rows))
    if nb_columns == 0:
        nb_columns = 1
    row_index = index // nb_columns
    column_index = index % nb_columns
    return row_index, column_index
        

class Experiment:
    #TODO : move option verbose in session class
    def __init__(self, name = None, win = None, mode=None) -> None:
        if mode not in MODES:
            raise ValueError(f"Mode {mode} not supported. Supported modes are {MODES}")
        else:
            self.mode = mode
        self.name = name
        self.running = False
        self.path = None       
        self.selected_session = None
        self.win = win 
        
    def set_path(self, path):
        path_exists = os.path.exists(path)
        if not path_exists:
            messagebox.showinfo("Experiment folder not found", f"Experiment folder not found in {path}, please check the path you wrote.")
        else:
            self.path = path
        return path_exists
        
    def fetch_sessions(self):
        if self.mode == 'Recording':
            self.sessions_indexes = range(1, len(SESSION_OPTIONS)+1)
            return SESSION_OPTIONS
        else:
            # list all folders from the experiment folder, directories only, begining with "Session_"
            self.session_folders = [f for f in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, f)) and f.startswith("Session_")]        
            self.session_folders.sort()
            # get the list of session indexes, e.g. [1, 2, 3]
            self.sessions_indexes = []
            self.sessions_options = []
            for session_folder in self.session_folders:
                try:
                    session_index = int(session_folder.split("_")[1])
                    self.sessions_indexes.append(session_index)
                    self.sessions_options.append(SESSION_OPTIONS[session_index - 1])
                except:
                    print(f"Session folder {session_folder} does not follow the naming convention 'Session_X', with X an integer")
            #TODO : return only sessions with participants
            # self.sessions = [Session(self.path, index,  mode = self.mode) for index in self.sessions_indexes]
            return self.sessions_options
    
    def set_session(self, selected_session):
        #find the index of the selected session in SESSION_OPTIONS
        index = SESSION_OPTIONS.index(selected_session)
        # self.selected_session = self.sessions[index]
        self.selected_session = Session(self.path, self.sessions_indexes[index], mode = self.mode)
        return self.selected_session
    
    def select_participant(self, pseudo):
        return self.selected_session.select_participant(pseudo)
    
    def process_selected_participants(self, process_labels):   
        if process_labels['Name'] == 'Replay':
            self.selected_session.replay_selected_participants()
        elif process_labels['Name'] == 'Pre-processing':
            self.selected_session.pre_process_selected_participants()
        elif process_labels['Name'] == 'Analysis':
            self.selected_session.analyse_selected_participants()
        else:
            print(f"Process {process_labels['Name']} not supported")
            # def get_participants(self):
    #     return self.selected_session.get_participants()
    
    def get_session_label(self):
        return self.selected_session.label
    
    def get_session_path(self):
        return self.selected_session.path
    
    def get_session_experimental_parameters(self):
        return self.selected_session.get_experimental_parameters()
    
    def set_session_experimental_parameters(self, parameters):
        self.selected_session.set_experimental_parameters(parameters)
    
    def get_session_recording_parameters(self):
        return self.selected_session.get_recording_parameters()
    
    def set_session_recording_parameters(self, parameters):
        self.selected_session.set_recording_parameters(parameters)
        self.selected_session.save_recording_parameters()
        
    def save_session_experimental_parameters(self):
        self.selected_session.save_experimental_parameters()
    
    def get_session_participants(self):
        return self.selected_session.get_participants() 
    
    def get_session_processing_monitoring(self):
        return self.selected_session.get_processing_monitoring()
    
    def get_pseudo(self, participant_firstname, participant_surname, location):
        return self.selected_session.get_pseudo(participant_firstname, participant_surname, location)
    
    def refresh_session(self):
        self.selected_session.import_participants_database()
        self.selected_session.import_pseudos_participants_database()
    
    def close(self):
        if self.selected_session is not None:
            self.selected_session.close()

class Session:
    _PARTICIPANTS_DATABASE_FILE_SUFFIX = "_participants_database.csv"
    _PARTICIPANTS_PSEUDOS_DATABASE_FILE_SUFFIX = "_participants_pseudos_database.csv"
    _EXPERIMENTAL_PARAMETERS_SUFFIX = "_experimental_parameters.csv"
    _RECORDING_PARAMETERS_SUFFIX = "_recording_parameters.csv"
    _INSTRUCTIONS_LANGUAGES_SUFFIX = "_instructions_languages.csv"
    _PROCESSING_MONITORING_SUFFIX = "_processing_monitoring.csv"
    
    _PARTICIPANTS_DATABASE_HEADER = ["Pseudo", "Date", "Handedness", "Location", "Number of trials"]
    _PROCESSING_MONITORING_HEADER = ["Pseudo", "Recording date", "Pre-processing date", "Replay date", "Analysis date", "Status", "Number of trials", "Number of trials pre-processed", "Number of trials replayed", "Number of trials analysed"]
    _PARTICIPANTS_PSEUDOS_DATABASE_HEADER = ["FirstName", "Surname", "Pseudo"]
    _SUPPORTED_LANGUAGES = ["French", "English"]
    _INSTRUCTIONS_LANGUAGES_HEADER = ["Label"] + _SUPPORTED_LANGUAGES
    _DEFAULT_PARAMETERS_LIST = ["Objects", "Hands", "Grips", "Movement Types", "Number of repetitions"]    #TODO : MODIFY THIS ACCORDING TO YOUR NEEDS

    def __init__(self, experiment_path, index, mode = 'Recording') -> None:
        self.index = index
        self.label = f"Session {self.index}"
        self.folder = f"Session_{self.index}"
        
        self.path = os.path.join(experiment_path, self.folder)  
        self.processing_path = os.path.join(experiment_path, f"{self.folder}_processing")
        self.pre_processing_path = os.path.join(self.processing_path, "Pre_processing")
        self.replay_path = os.path.join(self.processing_path, "Replay")
        self.analysis_path = os.path.join(self.processing_path, "Analysis")        
        self.evaluation_path = os.path.join(f'{self.processing_path}/Evaluation')
        self.plot_path = os.path.join(f'{self.processing_path}/Plots')
        
        self.mode = mode
        
        if mode in ['Pre-processing', 'Replay', 'Analysis']:
            if not os.path.exists(self.processing_path):
                answer = messagebox.askyesno(f"Processing folder not found", f"Processing folder not found at {self.processing_path}, please check the session folder. Do you want to create a new one?")
                if answer :
                    os.makedirs(self.processing_path)
                    print(f"New processing folder created")        
        if self.mode == 'Pre-processing':
            if not os.path.exists(self.processing_path):
                answer = messagebox.askyesno(f"Processing folder not found", f"Processing folder not found at {self.processing_path}, please check the session folder. Do you want to create a new one?")
                if answer :
                    os.makedirs(self.processing_path)
                    print(f"New pre-processing folder created")
        if self.mode == 'Replay':
            if not os.path.exists(self.replay_path):
                answer = messagebox.askyesno(f"Replay folder not found", f"Replay folder not found at {self.replay_path}, please check the session folder. Do you want to create a new one?")
                if answer :
                    os.makedirs(self.replay_path)
                    print(f"New replay folder created")
        if self.mode == 'Analysis':
            if not os.path.exists(self.analysis_path):
                answer = messagebox.askyesno(f"Analysis folder not found", f"Analysis folder not found at {self.analysis_path}, please check the session folder. Do you want to create a new one?")
                if answer :
                    os.makedirs(self.analysis_path)
                    print(f"New analysis folder created")
        
        self.all_participants = []
        self.current_participant = None
        self.participants_to_process = []   
        self.continue_processing = True     
        
        self.all_data_available = True
        self.missing_data = []
        self.preselect_all_participants = False
        
        self.is_new = not os.path.exists(self.path)
        self.experimental_parameters = None
        self.params_separators = [';', ','] #separator used in the experiment parameters csv file
        self.params_separator = self.params_separators[0] #default separator used in the experiment parameters csv file
        self.parameters_list = self._DEFAULT_PARAMETERS_LIST #list of parameters to be used in the experiment, can be modified by the user
        
        if self.is_new :
            if self.mode != 'Recording':
                messagebox.showinfo("Session folder not found", f"Session {self.label} folder not found in {experiment_path}, please check the main folder.")
                return False
            os.makedirs(self.path)
            self.participants_database = pd.DataFrame(columns=self._PARTICIPANTS_DATABASE_HEADER)
            self.participants_pseudos_database = pd.DataFrame(columns=self._PARTICIPANTS_PSEUDOS_DATABASE_HEADER)
            self.processing_monitoring_database = pd.DataFrame(columns=self._PROCESSING_MONITORING_HEADER)
        
        else:                
            print(f"Reading session {self.label} folder at {self.path}")
            self.import_participants_database()
            
            if self.participants_database is not None and self.mode != 'Recording':
                self.import_processing_monitoring()
                self.scan_participants_basic_data()
            self.import_pseudos_participants_database()
            self.import_instructions_languages()
            self.read_experimental_parameters()
            self.read_recording_parameters()
    
        print(f"Selected session: {self.label}")
        if self.mode != 'Recording':
            self.extract_devices_data()
            if not self.all_data_available:
                print(f"Session {self.label} incomplete. Missing data: {self.missing_data}")
                
        self.experiment_pre_processor = None
        self.experiment_replayer = None
        self.experiment_analyser = None
    
    def build_progress_display(self):
        name = "Processing..."
        self.progress_window = tk.Toplevel()
        self.progress_window.geometry("750x450")
        self.progress_window.title(name)
        label = ttk.Label(self.progress_window, text=name)
        label.pack()
        label = ttk.Label(self.progress_window, text="Please wait until the end of the process.")
        label.pack()
        
        messagebox.showinfo("Processing", "Processing started, please wait until the end of the process.")
        self.devices_progress_display = ProgressDisplay(len(self.devices_data), "devices", parent=self.progress_window, title="Devices") 
        self.devices_progress_display.pack(padx=10, pady=10)
        self.participants_progress_display= ProgressDisplay(len(self.participants_to_process), "participants pre-processed", parent=self.devices_progress_display, title = "Participants")
        self.participants_progress_display.pack(padx=10, pady=10)
        self.trials_progress_display= ProgressDisplay(self.participants_to_process[0].get_number_of_trials(), "trials pre-processed", parent=self.participants_progress_display, title = "Trials")
        self.trials_progress_display.pack(padx=10, pady=10)
        self.current_trial_progress_display= ProgressDisplay(self.participants_to_process[0].get_number_of_trials(), "trials pre-processed", parent=self.trials_progress_display, title = "Current Trial")
        self.current_trial_progress_display.pack(padx=10, pady=10)
        
        interrupt_button = ttk.Button(self.progress_window, text="Interrupt", command=self.interrupt_processing)
        interrupt_button.pack(padx=10, pady=10)
        
        self.progress_window.update()
        
    def read_experimental_parameters_new(self):
        #read the experiment parameters from the csv file
        parameters_path = os.path.join(self.path, f'{self.folder}{self._EXPERIMENTAL_PARAMETERS_SUFFIX}')
        if not os.path.exists(parameters_path):
            self.experimental_parameters = None
            print(f"Parameters file not found in {parameters_path}, please check the session folder.")
            messagebox.showinfo("Parameters file not found", f"{self.label} parameters file not found in {self.path}, please check the session folder.")
            self.all_data_available = False
            self.missing_data.append('experimental parameters')
        else:
            self.experimental_parameters = pd.read_csv(parameters_path)
            print(f"Experiment parameters read from '{parameters_path}'")
       
    def read_experimental_parameters(self):
        csv_path = os.path.join(self.path, f'{self.folder}{self._EXPERIMENTAL_PARAMETERS_SUFFIX}')
        if not os.path.exists(csv_path):
            self.experimental_parameters = None
            print(f"Experimental parameters file not found in {csv_path}, please check the session folder.")
            messagebox.showinfo("Experimental parameters file not found", f"{self.label} experimental parameters file not found in {self.path}, please check the session folder.")
            self.all_data_available = False
            self.missing_data.append('experimental parameters')
        else:
            self.experimental_parameters = {}
            reinit_reading = False
            with open(csv_path, "r") as csvfile:
                reader = csv.reader(csvfile, delimiter=self.params_separator)
                for row in reader:
                    param_type = row[0]
                    if self.params_separators[1] in param_type:
                        self.params_separator = self.params_separators[1]
                        reinit_reading = True
                        break
                    
                    param_list = row[1:]
                    #keep only non-empty parameters
                    param_list = [param for param in param_list if param != '']
                    self.experimental_parameters[param_type] = param_list
            if reinit_reading:
                self.read_experimental_parameters()
            else:
                print(f"Experimental parameters read from '{csv_path}'")
                self.parameters_list = list(self.experimental_parameters.keys())
    
    def set_experimental_parameters(self, parameters):
        self.experimental_parameters = parameters
        save_instructions = False
        for param_list in self.experimental_parameters.values():
            for param in param_list:
                #check if param is a blank string
                if param == '':
                    continue
                if param not in self.instructions_languages['Label'].values:
                    print(self.instructions_languages)
                    self.instructions_languages.loc[len(self.instructions_languages)] = [param]+[None for l in self._SUPPORTED_LANGUAGES]
                    for language in self._SUPPORTED_LANGUAGES:
                        self.ask_param_instructions(param, language)
                        save_instructions = True
        if save_instructions:
            self.save_instructions_languages()
                    
    def ask_param_instructions(self, param, language):                    
        self.instructions_window = tk.Toplevel()
        size = "700x200"
        self.instructions_window.geometry(size)
        requirements_label = ttk.Label(self.instructions_window, text=f"Parameter '{param}' was not found in our database {self.instructions_languages_csv_path}. \n Please enter the corresponding instructions for the language {language}")
        requirements_label.pack()
        instructions_entry = ttk.Entry(self.instructions_window)
        instructions_entry.pack()
        validate_button = ttk.Button(self.instructions_window, text="Validate", command=lambda: self.add_instructions(param, language, instructions_entry.get()))
        validate_button.pack()
        self.instructions_window.wait_window()
            
    def add_instructions(self, param, language, instructions):
        self.instructions_languages.loc[len(self.instructions_languages)] = [param, language, instructions]
        self.instructions_window.destroy()
    
    def read_recording_parameters(self):
        csv_path = os.path.join(self.path, f'{self.folder}{self._RECORDING_PARAMETERS_SUFFIX}')
        if not os.path.exists(csv_path):
            self.recording_parameters = None
            print(f"Recording parameters file not found in {csv_path}, please check the session folder.")
            messagebox.showinfo("Recording parameters file not found", f"{self.folder} recording parameters file not found in {self.path}, please check the session folder.")
            self.all_data_available = False
            self.missing_data.append('recording parameters')
        else:
            self.recording_parameters = {}
            with open(csv_path, "r") as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    param_type = row[0]
                    param_list = row[1:]
                    self.recording_parameters[param_type] = param_list
            print(f"Recording parameters read from '{csv_path}'")
    
    def set_recording_parameters(self, recording_parameters):
        self.recording_parameters = recording_parameters
    
    def import_pseudos_participants_database(self):
        pseudos_csv_path = os.path.join(self.path, f"{self.folder}{self._PARTICIPANTS_PSEUDOS_DATABASE_FILE_SUFFIX}")        
        if not os.path.exists(pseudos_csv_path):
            self.participants_pseudos_database = None
            print(f"No Pseudos-participant database found in {self.path}")
            answer = messagebox.askyesno(f"Pseudos-participants database not found", "Pseudos-participants not found, please check the session folder. Do you want to create a new one?")
            if answer :
                self.participants_pseudos_database = pd.DataFrame(columns=self._PARTICIPANTS_PSEUDOS_DATABASE_HEADER)
                self.participants_pseudos_database.to_csv(os.path.join(self.path, f"{self.folder}{self._PARTICIPANTS_PSEUDOS_DATABASE_FILE_SUFFIX}"), index=False)
                print(f"New Pseudos-participants database created")
            else:
                self.missing_data.append('pseudo-participant database')
            self.all_data_available = False
        else:
            self.participants_pseudos_database = pd.read_csv(pseudos_csv_path)
            print(f"Pseudos-participants database imported from '{pseudos_csv_path}'")
            print(f"Pseudos-participants database: \n{self.participants_pseudos_database}")
    
    def import_participants_database(self):
        participants_csv_path = os.path.join(self.path, f"{self.folder}{self._PARTICIPANTS_DATABASE_FILE_SUFFIX}")        
        if not os.path.exists(participants_csv_path):
            self.participants_database = None
            print(f"No participant database found in {self.path}")
            answer = messagebox.askyesno(f"Participants database not found", "Participants database not found, please check the session folder. Do you want to create a new one?")
            if answer :
                self.participants_database = pd.DataFrame(columns=self._PARTICIPANTS_DATABASE_HEADER)
                self.participants_database.to_csv(os.path.join(self.path, f"{self.folder}{self._PARTICIPANTS_DATABASE_FILE_SUFFIX}"), index=False)
                print(f"New participant database created")
            else:
                self.missing_data.append('participant database')
            self.all_data_available = False
        else:
            self.participants_database = pd.read_csv(participants_csv_path)
            print(f"Pseudos database imported from '{participants_csv_path}'")
            #add a column "To Process" to the database, with each row filled with True
            self.participants_database["To Process"] = self.preselect_all_participants
            
            
    def import_instructions_languages(self):
        self.instructions_languages_csv_path = os.path.join(self.path, f"{self.folder}{self._INSTRUCTIONS_LANGUAGES_SUFFIX}")
        if not os.path.exists(self.instructions_languages_csv_path):
            self.instructions_languages = None
            print(f"No instructions languages database found in {self.path}")
            answer = messagebox.askyesno(f"Instructions languages database not found", "Instructions languages database not found, please check the session folder. Do you want to create a new one?")
            if answer :
                self.instructions_languages = pd.DataFrame(columns=self._INSTRUCTIONS_LANGUAGES_HEADER)
                self.instructions_languages.loc[len(self.instructions_languages)] = ['Welcome',  "BIENVENUE DANS L'EXPERIENCE I-GRIP", "WELCOME TO THE I-GRIP EXPERIMENT"]
                self.save_instructions_languages()
                print(f"New instructions languages database created")
            else:
                self.missing_data.append('instructions languages database')
            self.all_data_available = False
        else:
            self.instructions_languages = pd.read_csv(self.instructions_languages_csv_path)
            print(f"Instructions languages database imported from '{self.instructions_languages_csv_path}'")
            
    def scan_participants_basic_data(self):
        for index, row in self.participants_database.iterrows():
            participant = Participant(row['Pseudo'], self.path, self.experimental_parameters, mode=self.mode)
            self.all_participants.append(participant)
            # check if the participant is pre-processed
            # self.participants_database.loc[index, 'Processed'] = participant.is_processed()
            self.participants_database.loc[index, 'All data available'] = participant.is_all_data_available()
            # self.participants_database.loc[index, 'Folder available'] = participant.is_folder_available()
            # self.participants_database.loc[index, 'Combinations available'] = participant.is_combinations_available()
            # self.participants_database.loc[index, 'All trial folders available'] = participant.is_all_trial_folders_available()
            self.processing_monitoring_database.loc[index, 'Pseudo'] = row['Pseudo']
            self.processing_monitoring_database.loc[index, 'Recording date'] = row['Date']
            self.processing_monitoring_database.loc[index, 'Status'] = participant.get_status()
            self.processing_monitoring_database.loc[index, 'Number of trials'] = participant.get_number_of_trials()
            self.processing_monitoring_database.loc[index, 'Number of trials pre-processed'] = participant.get_number_of_pre_processed_trials()
            self.processing_monitoring_database.loc[index, 'Number of trials replayed'] = participant.get_number_of_replayed_trials()
            self.processing_monitoring_database.loc[index, 'Number of trials analysed'] = participant.get_number_of_analysed_trials()
            self.processing_monitoring_database.loc[index, 'Processable'] = participant.is_folder_available() and participant.is_combinations_available() and participant.get_number_of_trials() > 0
            self.processing_monitoring_database.loc[index, 'To Process'] = False
        self.save_processing_monitoring()
            
    def import_processing_monitoring(self):
        monitoring_csv_path = os.path.join(self.processing_path, f"{self.folder}{self._PROCESSING_MONITORING_SUFFIX}")
        if not os.path.exists(monitoring_csv_path):
            print(f"No processing monitoring database found in {self.path}")
            answer = messagebox.askyesno(f"Processing monitoring database not found", "Processing monitoring database not found, please check the session folder. Do you want to create a new one?")
            if answer :
                self.processing_monitoring_database = pd.DataFrame(columns=self._PROCESSING_MONITORING_HEADER)
                self.processing_monitoring_database.to_csv(monitoring_csv_path, index=False)
                print(f"New processing monitoring database created")
        else:
            self.processing_monitoring_database = pd.read_csv(monitoring_csv_path)
            print(f"Processing monitoring database imported from '{monitoring_csv_path}'")
        
        self.processing_monitoring_database[ 'Processable'] = False
        self.processing_monitoring_database[ 'To Process'] = False
        print('Processing monitoring database: \n', self.processing_monitoring_database)
            
    def extract_devices_data(self):
        print("Extracting devices data...")
        #TODO : make sure that the devices data are available for all participants
        #list files with .npz extension in main path
        npz_files = [f for f in os.listdir(self.path) if f.endswith(".npz")]
        if len(npz_files) == 0:
            self.devices_data = None
            print(f"No device data found in {self.path}")
            self.missing_data.append("devices data")
        else:
            self.devices_data = {}
            #loop over all files
            for npz_file in npz_files:
                #get the device id from the file name
                device_id = npz_file.split("_")[1].split(".")[0]
                # extract the data from the file
                data = np.load(os.path.join(self.path, npz_file))
                self.devices_data[device_id] = data
            print(f"Devices data: {self.devices_data}")
    
    def get_experimental_parameters(self):
        return self.experimental_parameters
    
    def get_recording_parameters(self):
        return self.recording_parameters
    
    def get_participants(self):
        return self.participants_database
    
    def get_processing_monitoring(self):
        return self.processing_monitoring_database
    
    def select_participant(self, pseudo):
        # change the value at the line of pseudo and the column "To Process" to true
        bool = self.processing_monitoring_database.loc[self.processing_monitoring_database['Pseudo'] == pseudo, 'To Process'].values[0]
        self.processing_monitoring_database.loc[self.processing_monitoring_database['Pseudo'] == pseudo, 'To Process'] = not bool
        if not bool:
            print(f"Pseudo '{pseudo}' selected for pre-processing")
        else:
            print(f"Pseudo '{pseudo}' deselected for pre-processing")
        print(f"Processing database: \n{self.processing_monitoring_database}")
        nb_selected_participants = len(self.processing_monitoring_database.loc[self.processing_monitoring_database['To Process']==True])
        return nb_selected_participants
    
    def start(self):
        self.save_databases()
        self.current_participant.initiate_experiment()
        
    def fetch_participants_to_process(self):        
        self.continue_processing = True                
        for index, row in self.processing_monitoring_database.iterrows():
            if row['To Process']:
                self.participants_to_process.append(self.all_participants[index])
                
    def pre_process_selected_participants(self):
        self.fetch_participants_to_process()
        self.save_processing_monitoring()
        self.continue_processing = True
        print('Building experiment pre-processor...')
        self.experiment_pre_processor = epp.ExperimentPreProcessor()
        print('Experiment pre-processor built')
        print('Pre-processing selected participants...')
        for participant in self.participants_to_process:
            self.processing_monitoring_database.loc[self.processing_monitoring_database['Pseudo'] == participant.pseudo, 'Pre-processing date'] = pd.Timestamp.now()
            self.save_processing_monitoring()
            participant.pre_process(self.experiment_pre_processor)
            if self.continue_processing == False:
                break
        self.save_processing_monitoring()
        self.experiment_pre_processor.stop()
        
    def replay_selected_participants(self):    
        self.fetch_participants_to_process()
        self.build_progress_display()
        
        for device_id, device_data in self.devices_data.items():
            print(f"Building experiment replayer for device {device_id} with device_data: resolution {device_data['resolution']}, matrix {device_data['matrix']}")
            self.current_device_id = device_id
            self.experiment_replayer = erp.ExperimentReplayer(device_id, device_data)
            self.devices_progress_display.set_current(f"Processing device {device_id}")
            self.progress_window.update_idletasks()
            print("updating progress window")
            self.progress_window.update()
            print(f"Experiment replayer for device {device_id} built")
            # Loop over selected participants 
            for participant in self.participants_to_process:        
                self.processing_monitoring_database.loc[self.processing_monitoring_database['Pseudo'] == participant.pseudo, 'Replay date'] = pd.Timestamp.now()
                self.save_processing_monitoring()
                self.participants_progress_display.set_current(f"Processing participant {participant.pseudo}")
                self.progress_window.update()
                participant.set_progress_display( self.progress_window, self.trials_progress_display)
                participant.replay(self.experiment_replayer)
                self.participants_progress_display.increment()  
                      
                self.progress_window.update()
                if self.continue_processing == False:
                    break
            self.save_processing_monitoring()
            self.participants_progress_display.reset()
            self.devices_progress_display.increment()
            self.progress_window.update()
            self.experiment_replayer.stop()
            if self.continue_processing == False:
                break
            print(f"Experiment replayer for device {device_id} stopped")
        print("All selected participants replayed")
        # self.progress_window.destroy()
    
        
    def analyse_selected_participants(self):
        self.fetch_participants_to_process()
        self.build_progress_display()
        self.continue_processing = True
        self.evaluate()
        for device_id, device_data in self.devices_data.items():
            print(f"Building experiment analyser for device {device_id} with device_data: resolution {device_data['resolution']}, matrix {device_data['matrix']}")
            self.current_device_id = device_id
            self.experiment_analyser = ea.ExperimentAnalyser(device_id, device_data)
            self.devices_progress_display.set_current(f"Processing device {device_id}")
            self.progress_window.update_idletasks()
            print("updating progress window")
            self.progress_window.update()
            print(f"Experiment analyser for device {device_id} built")
            # Loop over selected participants 
            for participant in self.participants_to_process:     
                self.processing_monitoring_database.loc[self.processing_monitoring_database['Pseudo'] == participant.pseudo, 'Analysis date'] = pd.Timestamp.now()
                self.save_processing_monitoring()
                self.participants_progress_display.set_current(f"Analysing participant {participant.pseudo}")
                self.progress_window.update()
                participant.set_progress_display( self.progress_window, self.trials_progress_display)
                participant.analyse(self.experiment_analyser)
                self.participants_progress_display.increment()     
                      
                self.progress_window.update()
                if self.continue_processing == False:
                    break
            self.save_processing_monitoring()
            self.participants_progress_display.reset()
            self.devices_progress_display.increment()
            self.progress_window.update()
            self.experiment_analyser.stop()
            if self.continue_processing == False:
                break
            print(f"Experiment analyser for device {device_id} stopped")
        self.evaluate()
        print("All selected participants analysed")
    
    def evaluate(self, device_id=None):
        if device_id is None:
            summary_files = [f for f in os.listdir(self.evaluation_path) if f.endswith("summary.csv") and "global" not in f]
            global_summary_path = os.path.join(self.evaluation_path, "global_summary.csv")
        else:
            summary_files = [f for f in os.listdir(self.evaluation_path) if f.endswith("summary.csv") and device_id in f]
            global_summary_path = os.path.join(self.evaluation_path, f"global_summary_{device_id}.csv")
            
        if len(summary_files) == 0:
            print(f"No summary file found for device {device_id} in {self.evaluation_path}")
            return
        first_summary = pd.read_csv(os.path.join(self.evaluation_path, summary_files[0]))        
        metrics = first_summary.iloc[:, 0]
        
        global_evaluation_df = pd.DataFrame(columns=['Participant']+list(metrics))
        for i, summary_file in enumerate(summary_files):
            participant_and_device_id = summary_file.split("_summary")[0]
            summary = pd.read_csv(os.path.join(self.evaluation_path, summary_file))
            print(f"Summary for device {device_id}:\n{summary}")
            #get metrics that are in the first column, and the values in the second column, starting from the second row
            metrics = summary.iloc[:, 0]
            global_evaluation_df.loc[i, 'Participant'] = participant_and_device_id
            for j, metric in enumerate(metrics):
                global_evaluation_df.loc[i, metric] = summary.iloc[j, 1]
                
        for j, metric in enumerate(metrics):
            sum = global_evaluation_df[metric].sum()
            mean = global_evaluation_df[metric].mean()
            std = global_evaluation_df[metric].std()
            min = global_evaluation_df[metric].min()
            max = global_evaluation_df[metric].max()
            
            global_evaluation_df.loc['Sum', metric] = sum
            global_evaluation_df.loc['Mean', metric] = mean                                         
            global_evaluation_df.loc['Std', metric] = std
            global_evaluation_df.loc['Min', metric] = min
            global_evaluation_df.loc['Max', metric] = max
            
            global_evaluation_df.loc['Sum', 'Participant'] = 'Sum'
            global_evaluation_df.loc['Mean', 'Participant'] = 'Mean'
            global_evaluation_df.loc['Std', 'Participant'] = 'Std'
            global_evaluation_df.loc['Min', 'Participant'] = 'Min'
            global_evaluation_df.loc['Max', 'Participant'] = 'Max'
        
        #drop index
        global_evaluation_df = global_evaluation_df.reset_index(drop=True)
        # transpose the dataframe
        global_evaluation_df = global_evaluation_df.T
        
        
        #save the global summary withouth columns names
        global_evaluation_df.to_csv(global_summary_path, header=False)
                
        
    def interrupt_processing(self):
        print("Interrupting pre-processing...")
        self.continue_processing = False
        
    def is_data_available(self):
        return self.all_data_available
    
    def choose_existing_participant(self, participant_firstname, participant_surname):
        #TODO
        pass
    
    def get_participant(self, participant_firstname, participant_surname, handedness, location, language = 'English'):
        #check if the pseudo already exists
        pseudo_in_db = db.check_participant_in_database(participant_firstname, participant_surname, self.participants_pseudos_database)
        if not pseudo_in_db:
            pseudo = db.generate_new_random_pseudo(self.participants_pseudos_database)
            validate_pseudo = messagebox.askquestion("New Participant", f"New participant {participant_firstname} {participant_surname} created with pseudo {pseudo}. Do you want to validate this pseudo?")
            if validate_pseudo != 'yes':
                return self.get_participant(participant_firstname, participant_surname, handedness, location)
            #update the databases
            date = pd.Timestamp.now()
            print(f'database: {self.participants_database}')
            new_data =  [pseudo, date, handedness, location, 0, False]
            print(f"new_data: {new_data}")
            self.participants_database.loc[len(self.participants_database)] = new_data
            print(f"self.participants_pseudos_database ici: {self.participants_pseudos_database}")
            self.participants_pseudos_database.loc[len(self.participants_pseudos_database)] = [participant_firstname, participant_surname, pseudo]
            print(f"self.participants_pseudos_database là: {self.participants_pseudos_database}")
            print(f"New participant '{pseudo}' created")
            #TODO
        else:
            print(f"Participant {participant_firstname} {participant_surname} already exists")
            load_existing_participant = messagebox.askquestion("Existing Participant", f"Participant {participant_firstname} {participant_surname} already registered in the database, with pseudo {pseudo_in_db}. Do you want to load its data to complete it if needed?")
            if load_existing_participant != 'yes':
                return None
            else:
                pseudo = pseudo_in_db 
                print(f"Participant {participant_firstname} {participant_surname} selected to complete its trials")
        print(f'Session database counts now {len(self.participants_database)} participants')
        self.current_participant = Participant(pseudo, self.path, self.experimental_parameters, self.recording_parameters, mode=self.mode, language=language)
        self.current_participant.set_instructions(self.instructions_languages)
        return pseudo
    
    def save_databases(self):
        #remove the column "To Process" from the database  
        to_save = self.participants_database.drop(columns=['To Process'])
        to_save.to_csv(os.path.join(self.path, f"{self.folder}{self._PARTICIPANTS_DATABASE_FILE_SUFFIX}"), index=False)
        print(f'Participants database saved to {os.path.join(self.path, f"{self.folder}{self._PARTICIPANTS_DATABASE_FILE_SUFFIX}")}')
        print(f'Participants database: \n{self.participants_database}')
        self.participants_pseudos_database.to_csv(os.path.join(self.path, f"{self.folder}{self._PARTICIPANTS_PSEUDOS_DATABASE_FILE_SUFFIX}"), index=False)
        print(f'Participants pseudos database saved to {os.path.join(self.path, f"{self.folder}{self._PARTICIPANTS_PSEUDOS_DATABASE_FILE_SUFFIX}")}')
        print(f'Participants pseudos database: \n{self.participants_pseudos_database}')
    
    def save_processing_monitoring(self):
        self.processing_monitoring_database.to_csv(os.path.join(self.processing_path, f"{self.folder}{self._PROCESSING_MONITORING_SUFFIX}"), index=False)
        print(f'Processing monitoring database saved to {os.path.join(self.processing_path, f"{self.folder}{self._PROCESSING_MONITORING_SUFFIX}")}')
        print(f'Processing monitoring database: \n{self.processing_monitoring_database}')
        
    def save_experimental_parameters(self):
        csv_path = os.path.join(self.path, f'{self.folder}{self._EXPERIMENTAL_PARAMETERS_SUFFIX}')
        #check if the file already exists
        if os.path.exists(csv_path):
            overwrite = messagebox.askyesno("File already exists", f"File {csv_path} already exists. Do you want to overwrite it?")
            if overwrite:
                #copy the file to a backup, adding a timestamp
                csv_backup_path = os.path.join(self.path, f"bckp_experimental_parameters_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv")
                os.rename(csv_path, csv_backup_path)
                print('Parameters file backuped to {csv_backup_path}')
            else:
                return
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            params=[]
            for param_type, param_list in self.experimental_parameters.items():
                params.append([param_type]+param_list)
            writer.writerows(params,)
        print(f"Parameters written to '{csv_path}'")
        
    def save_recording_parameters(self):
        csv_path = os.path.join(self.path, f'{self.folder}{self._RECORDING_PARAMETERS_SUFFIX}')
        #check if the file already exists
        if os.path.exists(csv_path):
            overwrite = messagebox.askyesno("File already exists", f"File {csv_path} already exists. Do you want to overwrite it?")
            if overwrite:
                #copy the file to a backup, adding a timestamp
                csv_backup_path = os.path.join(self.path, f"bckp_recording_parameters_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv")
                os.rename(csv_path, csv_backup_path)
                print('Parameters file backuped to {csv_backup_path}')
            else:
                return
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            params=[]
            for param_type, param_list in self.recording_parameters.items():
                params.append([param_type]+param_list)
            writer.writerows(params,)
        print(f"Parameters written to '{csv_path}'")
        
    def save_instructions_languages(self):        
        self.instructions_languages.to_csv(os.path.join(self.path, f"{self.folder}{self._INSTRUCTIONS_LANGUAGES_SUFFIX}"), index=False)
    
    def close(self):
        if self.experiment_replayer is not None:
            self.experiment_replayer.stop()
        if self.experiment_pre_processor is not None:
            self.experiment_pre_processor.stop()
        
        if self.current_participant is not None:
            self.current_participant.close()
            
class Participant:
    def __init__(self, pseudo, session_path, session_experimental_parameters=None, recording_parameters=None, mode = 'Recording', language='English') -> None:
        self.pseudo = pseudo
        self.experimental_parameters = session_experimental_parameters
        self.recording_parameters = recording_parameters
        self.combinations_data = []
        self.session_path = session_path
        self.mode = mode
        self.combinations_data = None
        self.found_trial_folders = None
        self.current_trial_index = 0
        self.all_data_available = True
        self.processed = False
        self.missing_trial_folders = []
        self.missing_data = []
        self.available_trials = []
        self.missing_trials = []
        self.replayable_trials = []
        self.analyzable_trials = []
        self.expe_recorders = []
        self.language = language        
        self.trial_ongoing = False
        self.display_thread=None
        
        self.path = os.path.join(self.session_path, self.pseudo)
        self.pre_processing_path = os.path.join(f'{self.session_path}_processing/Pre_processing', self.pseudo)
        self.replay_path = os.path.join(f'{self.session_path}_processing/Replay', self.pseudo)
        self.analyse_path = os.path.join(f'{self.session_path}_processing/Analysis', self.pseudo)
        self.evaluation_path = os.path.join(f'{self.session_path}_processing/Evaluation')
        self.plot_path = os.path.join(f'{self.session_path}_processing/Plots', self.pseudo)
        
        if self.mode == 'Pre-processing':
            self.source_path = self.path
            self.destination_path = self.pre_processing_path
        elif self.mode == 'Replay':
            self.source_path = self.pre_processing_path
            self.destination_path = self.replay_path
        elif self.mode == 'Analysis':
            self.source_path = self.replay_path
            self.destination_path = self.analyse_path
        
        if self.mode == 'Pre-processing':
            if not os.path.exists(self.pre_processing_path):
                answer = messagebox.askyesno(f"Participant pre_processing folder not found", f"Participant processing folder not found in {self.pre_processing_path}. Do you want to create a new one?")
                if answer:
                    os.makedirs(self.pre_processing_path)
                    print(f"New participant pre_processing folder created in {self.pre_processing_path}")
        if self.mode == 'Replay':
            if not os.path.exists(self.replay_path):
                answer = messagebox.askyesno(f"Participant replay folder not found", f"Participant replay folder not found in {self.replay_path}. Do you want to create a new one?")
                if answer:
                    os.makedirs(self.replay_path)
                    print(f"New participant replay folder created in {self.replay_path}")   
        if self.mode == 'Analysis':
            if not os.path.exists(self.analyse_path):
                answer = messagebox.askyesno(f"Participant analysis folder not found", f"Participant analysis folder not found in {self.analyse_path}. Do you want to create a new one?")
                if answer:
                    os.makedirs(self.analyse_path)
                    print(f"New participant analysis folder created in {self.analyse_path}")     
            if not os.path.exists(self.plot_path):
                answer = messagebox.askyesno(f"Participant plot folder not found", f"Participant plot folder not found in {self.plot_path}. Do you want to create a new one?")
                if answer:
                    os.makedirs(self.plot_path)
                    print(f"New participant plot folder created in {self.plot_path}")
        
        self.combinations_path = os.path.join(self.path, f"{self.pseudo}_combinations.csv")
        self.data_csv_path = os.path.join(self.path, f"{self.pseudo}_data.csv")
        
        self.is_new = not os.path.exists(self.path)
        self.participant_window = None
        self.experimentator_window = None
        self.progress_window = None
        
        if self.is_new:     
            if mode != 'Recording':
                print(f"Participant folder not found")
                messagebox.showinfo(f"Participant folder not found", f"Participant folder not found in {self.path}")
                self.all_data_available = False
                self.missing_data.append('participant folder')
            else:
                os.makedirs(self.path)
                self.generate_combinations()
        
        if not self.is_new:
            self.get_combinations()
            self.fetch_trial_folders()       
            self.scan_found_trials()    
            if mode != 'Recording':
                self.check_processed()   
            else:
                if self.combinations_data is not None and len(self.available_trials)>0:
                    answer = messagebox.askyesnocancel(f"Participant folder already exists", f"Participant folder already exists in {self.path}. A combinations file and {len(self.available_trials)} trial folders were found. Please check the participant folder. Press 'yes' to resume the recording and complete missing trials. Press 'no' to delete existing data, generate a new combinations file and start a new recording. Else, press 'cancel' and select another participant.")
                    if answer == True:
                        self.all_data_available = True
                    elif answer == False:
                        self.back_up_data()
                        os.makedirs(self.path)
                        self.generate_combinations()
                    else:
                        return
                
    def back_up_data(self):
        if os.path.exists(self.path):
            #copy the folder to a backup, adding a timestamp
            backup_path = os.path.join(self.session_path, f"bckp_{self.pseudo}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}")
            os.rename(self.path, backup_path)
            print(f"Participant folder backuped to {backup_path}")
            
    
    def generate_combinations(self):
        # get the parameters from the session and number of repetitions separately
        number_of_repetitions = int(self.experimental_parameters['Number of repetitions'][0])
        parameters_list = [value for key, value in self.experimental_parameters.items() if key != 'Number of repetitions']
        keys_list = [key for key, value in self.experimental_parameters.items() if key != 'Number of repetitions']

        # Check if any of the lists is empty or contains only empty strings
        if any(not lst or all(val.strip() == "" for val in lst) for lst in parameters_list):
            messagebox.showinfo("Empty List", "One or more lists are empty or contain only blank strings. Please provide valid input.")
            return

        # Generate all combinations of the elements from the lists
        combinations_list = []
        for i in range(number_of_repetitions):
            combinations_list += list(itertools.product(*parameters_list))
        # Shuffle the combinations in random order
        random.shuffle(combinations_list)
        self.combinations_data = pd.DataFrame(combinations_list, columns=keys_list)
        self.missing_trials = []
        for index, row in self.combinations_data.iterrows():
            trial_folder_name = f"trial_{index}_combi"
            for key, value in row.items():
                trial_folder_name += f"_{value}"
            self.combinations_data.loc[index, 'Trial Folder'] = trial_folder_name
            self.combinations_data.loc[index, 'Trial Number'] = int(index+1)
            self.missing_trials.append(Trial(trial_folder_name, self.path, row, participant_pre_processing_path=self.pre_processing_path, participant_replay_path=self.replay_path, participant_analysis_path=self.analyse_path))
        self.save_combinations()
        print(f"{len(self.combinations_data)} combinations generated and saved to '{self.combinations_path}'")
            
    def get_combinations(self):
        # check if the combinations file exists
        if not os.path.exists(self.combinations_path):
            print(f"Combinations file not found in {self.combinations_path}")
            # messagebox.showinfo(f"Combinations file not found", f"Combinations file not found in {self.combinations_path}")
            self.all_data_available = False
            self.missing_data.append('combinations')
            self.combinations_data = None
        else:
            #create a pandas dataframe from the csv and read header from file            
            self.combinations_data = pd.read_csv(self.combinations_path)
            self.combinations_data = self.combinations_data.astype({'Trial Number': int})
            # for index, row in self.combinations_data.iterrows():
            #     trial_index = row['Trial Number']
            #     objects = row['Objects']
            #     hand = row['Hands']
            #     grip = row['Grips']
            #     movement_type = row['Movement Types']
            #     trial_folder_name = f"trial_{trial_index}_combi_{objects}_{hand}_{grip}_{movement_type}"
            #     self.combinations_data.loc[index, 'Trial Folder'] = trial_folder_name
            #     self.combinations_data.loc[index, 'Trial Number'] = int(trial_index+1)
            print(f"combinations data: \n{self.combinations_data}")
            print(f"Combinations read from '{self.combinations_path}'")
        
    def build_recorders(self, devices_ids, resolution, fps):
        print('LESSGOOOOOOO')
        for device_id in devices_ids:
            expe_recorder = erc.ExperimentRecorder(self.path, device_id = device_id, resolution = resolution, fps = fps)
            self.expe_recorders.append(expe_recorder)
    
    def initiate_experiment(self):
        devices_ids = self.recording_parameters['devices_ids']
        self.resolution = self.recording_parameters['resolution']
        fps = self.recording_parameters['fps'][0]
        self.build_UIs()
        self.build_recorders(devices_ids, self.resolution, fps)   
        self.save_experimental_parameters()
        self.save_recording_parameters()   
        
    def start_experiment(self):
        self.trial_ongoing = False
        self.start_button.state(['disabled'])
        self.display_next_trial_button.state(['!disabled'])
        self.stop_button.state(['!disabled'])
        self.current_trial_index=0
        for expe_recorder in self.expe_recorders:
            expe_recorder.init()
        self.expe_running=True
        self.display_thread = threading.Thread(target=self.display_task)
        self.display_thread.start()  
        
    def stop_experiment(self):
        print("Stopping experiment")
        if self.trial_ongoing:
            self.stop_current_trial()
        self.expe_running=False
        for rec in self.expe_recorders:
            rec.stop()
        if self.display_thread is not None:
            self.display_thread.join()
        print("Experiment stopped")     
        if self.participant_window is not None:
            self.participant_window.destroy()
        if self.experimentator_window is not None:
            self.experimentator_window.destroy()   

    def display_next_trial(self):
        self.current_trial = self.missing_trials[self.current_trial_index]
        self.trials_advancement.set(f"Trial {self.current_trial_index+1}/{self.nb_missing_trials}")
        self.current_trial_combination.set(self.current_trial.get_combination())
        procede = self.current_trial.check_and_make_dir()
        if procede:
            # self.instructions_text.set(self.current_trial.get_instructions())
            txt = self.current_trial.get_instructions_colored()
            self.instructions_text_widget.delete('1.0', tk.END)
            for text, tag in txt:
                self.instructions_text_widget.insert(tk.END, text, tag)
            self.display_next_trial_button.state(['disabled'])
            self.start_next_trial_button.state(['!disabled'])
        else:     
            self.current_trial_index += 1
            self.display_next_trial()
    
    def start_next_trial(self):
        self.trial_ongoing = True
        self.start_next_trial_button.state(['disabled'])
        self.stop_current_trial_button.state(['!disabled'])
        self.instructions_text_widget.insert(tk.END, " \n \n \n \nRecording", "recording")
        for rec in self.expe_recorders:
            rec.record_trial(self.current_trial)  

    def stop_current_trial(self):
        self.trial_ongoing = False
        for rec in self.expe_recorders:
            rec.stop_record()
        if self.current_trial_index >= len(self.missing_trials)-1:
            self.instructions_text_widget.delete('1.0', tk.END)
            self.instructions_text_widget.insert(tk.END, "\n \n \n \n \n \n CONGRATULATIONS, YOU HAVE COMPLETED THE EXPERIMENT !", "center")
        else:
            self.current_trial_index += 1
            self.display_next_trial_button.state(['!disabled'])
            self.display_next_trial()
        self.stop_current_trial_button.state(['disabled'])
        print(f"Stopped {self.current_trial.label}")
        
    def display_task(self):
        rotate = True
        while self.expe_running:
            imgs=[]
            for rec in self.expe_recorders:
                if rec.img is not None:
                    named_img = rec.img.copy()
                    if rotate:
                        named_img = cv2.rotate(named_img, cv2.ROTATE_90_CLOCKWISE)
                    #TODO : resize the image to fit the screen
                    named_img = cv2.putText(named_img, f'view {rec.device_id}', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (88, 205, 54), 1, cv2.LINE_AA)
                    if self.trial_ongoing:
                        named_img = cv2.putText(named_img, 'Recording', (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (10, 10, 255), 1, cv2.LINE_AA)
                    imgs.append(named_img)
            if len(imgs) > 0:
                side_by_side_img = cv2.hconcat(imgs)
                # cv2.imshow(self.current_trial.label,side_by_side_img)
                if not rotate:
                    side_by_side_img = cv2.resize(side_by_side_img, (self.resolution[0], int(self.resolution[1]/len(imgs))))
                cv2.imshow(f'Recording participant {self.pseudo}',side_by_side_img)
            k = cv2.waitKey(1)
            if k == ord('q'):
                break        
        cv2.destroyAllWindows()

    def build_participant_UI(self):
        self.participant_window = tk.Toplevel()
        size=720
        # self.participant_window.attributes('-fullscreen', True)
        self.participant_window.title(f'Instructions for participant {self.pseudo}')
        frame = ttk.Frame(self.participant_window)
        frame.pack(fill=tk.BOTH, expand=True)
        # self.instructions_text = tk.StringVar()
        # get the instruction corresponding to the label 'Welcome'
        # self.instructions_text.set(self.instructions.loc[self.instructions['Label'] == 'Welcome', 'Instructions'].values[0])
        #TODO : add language selection
        # self.instructions_text.set("Please read the instructions below and click on 'Start' when you are ready to start the experiment.")
        # self.instructions_label = ttk.Label(self.participant_window, textvariable=self.instructions_text, font=("Helvetica", 25), wraplength=size-20, justify='center')
        # #center the label vertically
        # self.instructions_label.pack(fill=tk.BOTH, expand=True)
        main_font = ["Helvetica", 30]
        
        self.instructions_text_widget = tk.Text(frame, font=("Helvetica", 25), wrap=tk.WORD)
        self.instructions_text_widget.pack(fill=tk.BOTH, expand=True,anchor='center')
        self.instructions_text_widget.tag_configure("center", justify='center',font=("Helvetica", 25, "bold"))
        self.instructions_text_widget.tag_configure("recording", justify='center',font=("Helvetica", 25, "bold"), foreground="red")
        self.instructions_text_widget.tag_configure("left", justify='left')
        self.instructions_text_widget.tag_configure("red", foreground="red")
        self.instructions_text_widget.tag_configure("green", foreground="green")
        self.instructions_text_widget.tag_configure("blue", foreground="blue")
        self.instructions_text_widget.tag_configure("purple", foreground="pink")
        self.instructions_text_widget.tag_configure("title", font=('Helvetica', 35), justify='center')
        self.instructions_text_widget.tag_configure("normal", font=main_font)
        self.instructions_text_widget.tag_configure("intro", font=main_font+["bold"], justify='center')
        self.instructions_text_widget.tag_configure("hand", font=main_font+["bold"], foreground="#5bc0de")
        self.instructions_text_widget.tag_configure("mov_type", font=main_font+["bold"], foreground="#5cb85c")
        self.instructions_text_widget.tag_configure("grip", font=main_font+["bold"], foreground="#ffc107")
        self.instructions_text_widget.tag_configure("object", font=main_font+["bold"], foreground="#d9534f")
        self.instructions_text_widget.tag_configure("bold", font=("Helvetica", 25, "bold"))
        self.instructions_text_widget.tag_configure("italic", font=("Helvetica", 25, "italic"))
        self.instructions_text_widget.tag_configure("underline", font=("Helvetica", 25, "underline"))
        self.instructions_text_widget.insert(tk.END, self.instructions.loc[self.instructions['Label'] == 'Welcome', 'Instructions'].values[0], "center")
    
    def build_experimentator_UI(self):
        self.experimentator_window = tk.Toplevel()
        self.experimentator_window.geometry("720x720")
        self.experimentator_window.title(f'Instructions for experimentator {self.pseudo}')
        frame = ttk.Frame(self.experimentator_window)
        frame.pack()
        # frame.pack(fill=tk.BOTH, expand=True)
        self.nb_missing_trials = len(self.missing_trials)
        self.trials_advancement = tk.StringVar()
        self.trials_advancement.set(f"Trial -/{self.nb_missing_trials}")
        self.trial_label = ttk.Label(frame, textvariable=self.trials_advancement, font=("Helvetica", 25), justify='center')
        self.trial_label.pack(fill=tk.BOTH, expand=True)
        self.current_trial_combination = tk.StringVar()
        self.current_trial_combination.set(f"Combination -")
        self.trial_combination_label = ttk.Label(frame, textvariable=self.current_trial_combination, font=("Helvetica", 25), justify='center')
        self.trial_combination_label.pack(fill=tk.BOTH, expand=True, pady=30)
        self.start_button = ttk.Button(frame, text="Start experiment", command=self.start_experiment, style='primary.TButton')
        self.start_button.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.display_next_trial_button = ttk.Button(frame, text="Display next trial", command=self.display_next_trial, style='secondary.TButton')
        self.display_next_trial_button.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.start_next_trial_button = ttk.Button(frame, text="Start trial", command=self.start_next_trial, style = 'success.TButton')
        self.start_next_trial_button.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.stop_current_trial_button = ttk.Button(frame, text="Stop current trial", command=self.stop_current_trial, style = 'warning.TButton')
        self.stop_current_trial_button.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.stop_button = ttk.Button(frame, text="Stop experiment", command=self.stop_experiment, style = 'danger.TButton')
        self.stop_button.pack(fill=tk.BOTH, expand=True, pady=30)
        self.display_next_trial_button.state(['disabled'])
        self.start_next_trial_button.state(['disabled'])
        self.stop_current_trial_button.state(['disabled'])
        self.stop_button.state(['disabled'])
    
    def build_UIs(self):
        self.build_experimentator_UI()
        self.build_participant_UI()
    
    def fetch_trial_folders(self):
        if self.combinations_data is None:            
            self.all_data_available = False
            self.missing_data.append('trial folders')
            return
        # list all folders from the participant folder, directories only
        self.found_trial_folders = [f for f in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, f))]
        print(f"Trial folders found in '{self.path}': \n{self.found_trial_folders}")
        # convert this list to a dataframe with a column named "Trial Folder"
        # self.found_trial_folders = pd.DataFrame(self.found_trial_folders, columns=['Trial Folder'])      
        # print(f"Trial folders read from '{self.path}': \n{self.found_trial_folders}")     
        #TODO
        # for trial_folder in self.combinations_data['Trial Folder']:
        for index, row in self.combinations_data.iterrows():
            # check if the trial_folder is in the list of found_trial_folders
            trial_folder = row['Trial Folder']
            # print(f"Checking trial folder '{trial_folder}'")
            # print(trial_folder)
            # print(self.found_trial_folders[0])
            # print(trial_folder == self.found_trial_folders[0])
            # print(trial_folder in self.found_trial_folders)
            if not trial_folder in self.found_trial_folders:
                self.missing_trial_folders.append(trial_folder)
                self.missing_trials.append(Trial(trial_folder, self.path, row, participant_pre_processing_path=self.pre_processing_path, participant_replay_path=self.replay_path, participant_analysis_path=self.analyse_path))
                # print(f"Trial folder '{trial_folder}' not found")
            else:
                self.available_trials.append(Trial(trial_folder, self.path, row, participant_pre_processing_path=self.pre_processing_path, participant_replay_path=self.replay_path, participant_analysis_path=self.analyse_path))
                # print(f"Trial folder '{trial_folder}' found")
            # if index == 1:
            #     klvm
        if len(self.missing_trial_folders) > 0:
            print(f"Participant '{self.pseudo}' missing {len(self.missing_trial_folders)} trial folders: ")
            # print(f"Participant '{self.pseudo}' missing trial folders: {self.missing_trial_folders}")
            self.all_data_available = False
            self.missing_data.append('trial folders')

        # self.missing_trials = [Trial(trial_folder, self.path) for trial_folder in self.missing_trial_folders]
        
    def scan_found_trials(self):
        self.nb_pre_processed_trials = 0
        self.nb_replayed_trials = 0
        self.nb_analysed_trials = 0
        self.status = 'Not processed'
        for trial in self.available_trials:
            if trial.was_pre_processed():
                self.nb_pre_processed_trials += 1
                self.replayable_trials.append(trial)
            if trial.was_replayed():
                self.nb_replayed_trials += 1
                self.analyzable_trials.append(trial)
            if trial.was_analysed():
                self.nb_analysed_trials += 1
                
        if self.is_folder_available() and self.is_combinations_available() and self.get_number_of_trials() > 0 :
            self.status = 'Ready to pre-process'
        
        if self.nb_pre_processed_trials == len(self.available_trials):
            self.status = 'Pre-processed'
        elif self.nb_pre_processed_trials > 0:
            self.status = 'Partially pre-processed'
        if self.nb_replayed_trials == len(self.available_trials):
            self.status = 'Replayed'
        elif self.nb_replayed_trials > 0:
            self.status = 'Partially replayed'
        if self.nb_analysed_trials == len(self.available_trials):
            self.status = 'Analysed'
        elif self.nb_analysed_trials > 0:
            self.status = 'Partially analysed'
                
    def get_number_of_pre_processed_trials(self):
        return self.nb_pre_processed_trials
    
    def get_number_of_replayed_trials(self):
        return self.nb_replayed_trials
    
    def get_number_of_analysed_trials(self):
        return self.nb_analysed_trials
    
    def get_status(self):
        return self.status
    
    def pre_process(self, experiment_pre_processor):
        experiment_pre_processor.set_new_participant(self.pseudo, len(self.available_trials))
        print( f"Pre-processing pseudo '{self.pseudo}'")
        check_path = os.path.join(self.pre_processing_path, f"{self.pseudo}_trials_check.csv")
        if os.path.exists(check_path):
            trials_check = pd.read_csv(check_path)
        else:
            trials_check = self.combinations_data
            trials_check['Combination OK'] = False
            trials_check['Face OK'] = False
            trials_check['Trial_duration'] = 0
        for i, trial in enumerate(self.available_trials):
            if trial.was_pre_processed():
                answer = messagebox.askyesnocancel(f"Trial already pre-processed", f"Trial {trial.label} already pre-processed. Do you want to re-process it?")
                if answer != True:
                    experiment_pre_processor.skip_trial()
                    continue
            print(f"Pre-processing trial {i}/{len(self.available_trials)}")
            combi_ok, face_ok, durations = trial.pre_process(experiment_pre_processor)
            print(f'combi_ok: {combi_ok}, face_ok: {face_ok}, durations: {durations}')
            trials_check.loc[trials_check['Trial Folder'] == trial.label, 'Combination OK'] = combi_ok
            trials_check.loc[trials_check['Trial Folder'] == trial.label, 'Face OK'] = face_ok
            trials_check.loc[trials_check['Trial Folder'] == trial.label, 'Stand_duration'] = durations['stand']
            trials_check.loc[trials_check['Trial Folder'] == trial.label, 'Movement_duration'] = durations['movement']
            trials_check.loc[trials_check['Trial Folder'] == trial.label, 'Contact_duration'] = durations['contact']
            trials_check.loc[trials_check['Trial Folder'] == trial.label, 'Return_duration'] = durations['return']
            trials_check.loc[trials_check['Trial Folder'] == trial.label, 'Trial_duration'] = durations['total']
            trials_check.to_csv(check_path, index=False)
            print(f'saved to {check_path}')
                
    def replay(self, experiment_replayer):
        
        print( f"Replaying pseudo '{self.pseudo}'")
        # create a dict to store 'Trial_duration' and 'Trial_data_extration_duration' for each trial
        trials_meta_data = ['Found', 'Trial_duration', 'Trial_data_extration_duration']
        # create an empty dataframe to store the trials_meta_data, same row index as self.combinations
        trials_meta_data_df = pd.DataFrame(columns=trials_meta_data, index=self.combinations_data.index)
        trials_meta_data_df['Found'] = False
        # concatenate the two dataframes into a dataframe self.data
        self.data = pd.concat([self.combinations_data, trials_meta_data_df], axis=1)
        #loop over trials
        global_answer = None
        replayer_ID = experiment_replayer.get_device_id()
        for trial in self.replayable_trials:          
            if trial.was_replayed(device_ID=replayer_ID):
                if global_answer is None:
                    answer = messagebox.askyesno(f"Trial already pre-processed", f"Trial {trial.label} already replayed for device {replayer_ID}. Do you want to re-process it?")
                    global_apply = messagebox.askyesnocancel("Do you want to apply this answer to all trials?", f"Do you want to apply this answer to all trials for device {replayer_ID} ?")
                    if global_apply:
                        global_answer = answer
                else:
                    answer = global_answer
                if answer != True:
                    continue
            self.progress_display.set_current(f"Replaying trial {trial.label}")  
            self.progress_window.update()
            trial_meta_data = trial.replay(experiment_replayer)
            self.progress_display.increment()
            self.progress_window.update()
            # put True in the 'Found' column of the self.data dataframe
            self.data.loc[self.data['Trial Folder'] == trial.label, 'Found'] = True
            
            for key in trial_meta_data.keys():
                self.data.loc[self.data['Trial Folder'] == trial.label, key] = trial_meta_data[key]
        
        #write the participant data to a csv file
        self.data.to_csv(self.data_csv_path, index=False)
        
        print(f"Processed pseudo '{self.pseudo}'")
        print(f"Participant '{self.pseudo}' missing trial {len(self.missing_trial_folders)} folders ")
    
    def analyse(self, experiment_analyser, do_analyse=True, evaluate=False, plot=False):
        print( f"Analyzing pseudo '{self.pseudo}'")
        global_answer = None
        analyser_ID = experiment_analyser.get_device_id()
        
        if do_analyse:
            evaluation_dict = {}
            for trial in self.analyzable_trials:
                if trial.was_analysed(device_ID=analyser_ID):
                    if global_answer is None:
                        answer = messagebox.askyesnocancel(f"Trial already analysed", f"Trial {trial.label} already analysed. Do you want to re-analyse it?")
                        global_apply = messagebox.askyesnocancel("Do you want to apply this answer to all trials?", f"Do you want to apply this answer to all trials for device {analyser_ID} ?")
                        if global_apply:
                            global_answer = answer
                    else:
                        answer = global_answer
                    if answer != True:
                        continue
                print(f"Analyzing trial {trial.label}")
                self.progress_display.set_current(f"Analyzing trial {trial.label}")  
                self.progress_window.update()
                trial_evaluation = trial.analyse(experiment_analyser, evaluate=evaluate)
                self.progress_display.increment()
                self.progress_window.update()
                if trial_evaluation is not None:
                    evaluation_dict[trial.label] = trial_evaluation
                
        if evaluate:
            self.evaluate(analyser_ID)
        
        if plot:
            self.plot_stuff(analyser_ID)
    
    def plot_stuff(self, device_ID):
        # list all files in the analysis folder
        analysis_files = [f for f in os.listdir(self.analyse_path) if os.path.isfile(os.path.join(self.analyse_path, f)) and f.endswith('.csv') and device_ID in f]
        if len(analysis_files) == 0:
            print(f"No analysis files found in '{self.analyse_path}'")
            return
        #order the files by label
        analysis_files.sort()
        for analysis_file in analysis_files:
            label = analysis_file.split('_target')[0]
            hand = label.split('_')[4]
            target = label.split('_')[3]
            mode = label.split('_')[6]
            target_data = pd.read_csv(os.path.join(self.analyse_path, analysis_file))
            print(f"Ploting trial {label}")
            # define plot path from the analysis path, replacing _target_data.csv by _plot.png
            plot_path = os.path.join(self.plot_path, f"{label}_plot.eps")
            Trial.plot_target_data(target_data, hand, plot_path, device_ID, target,mode)
            
    def evaluate(self, device_ID):
        # read the trials_check file
        check_path = os.path.join(self.pre_processing_path, f"{self.pseudo}_trials_check.csv")
        if os.path.exists(check_path):
            trials_check_df = pd.read_csv(check_path)
        else:                                                                    
            print(f"Trials check file not found in '{check_path}'")
            trials_check_df=None
        
        # list all files in the analysis folder
        analysis_files = [f for f in os.listdir(self.analyse_path) if os.path.isfile(os.path.join(self.analyse_path, f)) and f.endswith('.csv') and device_ID in f]
        if len(analysis_files) == 0:
            print(f"No analysis files found in '{self.analyse_path}'")
            return

        
        #order the files by label
        analysis_files.sort()
        evaluation_dict = {}
        for analysis_file in analysis_files:
            label = analysis_file.split('_cam')[0]
            hand = label.split('_')[4]
            grip = label.split('_')[5]
            target = label.split('_')[3]
            mode = label.split('_')[6]
            target_data = pd.read_csv(os.path.join(self.analyse_path, analysis_file))
            print(f"Evaluating trial {label}")
            eval_data = Trial.evaluate_target_data(target_data, device_ID=device_ID, hand=hand, target=target, mode=mode, grip=grip)
            #find the row in the trials_check_df corresponding to the label and the corresponding value in column 'Combination OK'
            if eval_data is not None:
                if trials_check_df is not None:
                    combi_ok = trials_check_df.loc[trials_check_df['Trial Folder'] == label, 'Combination OK'].values[0]
                    face_ok = trials_check_df.loc[trials_check_df['Trial Folder'] == label, 'Face OK'].values[0]
                    eval_data['combi_ok'] = combi_ok
                    eval_data['face_ok'] = face_ok
                evaluation_dict[label] = eval_data

        evaluation_df = pd.DataFrame.from_dict(evaluation_dict, orient='index')
        df_path = os.path.join(self.evaluation_path, f"{self.pseudo}_{device_ID}_evaluation.csv")
        evaluation_df.to_csv(df_path, index=True)
        
        nb_trials = len(analysis_files)
        nb_cam_same=0
        nb_cam_opposite=0
        nb_executed=0
        nb_simulated=0
        for index, row in evaluation_df.iterrows():
            if row['cam_hand_position']=='same':
                nb_cam_same += 1
            if row['cam_hand_position']=='opposite':
                nb_cam_opposite += 1
            if row['movement_mode'] == 'executed':
                nb_executed += 1
            if row['movement_mode'] == 'simulated':
                nb_simulated += 1
        # keep only valid trials and combi_ok
        evaluation_df = evaluation_df[evaluation_df['is_trial_valid'] & evaluation_df['combi_ok']]
        
        
        expected_objects = sc.RigidObject.LABEL_EXPE_NAMES.values()
        grips = ['palmar', 'pinch']
        nb_trials_valid_by_objects = {}
        for obj in expected_objects:
            nb_trials_valid_by_objects['nb_trials_valid_for_'+obj] = evaluation_df[evaluation_df['task_target'] == obj]['is_trial_valid'].sum()
            
        nb_trials_valid_by_grip = {}
        for grip in grips:
            nb_trials_valid_by_grip['nb_trials_valid_for_'+grip] = evaluation_df[evaluation_df['task_grip'] == grip]['is_trial_valid'].sum()

        nb_target_found_by_objects = {}
        for obj in expected_objects:
            nb_target_found_by_objects['nb_target_found_'+obj] = evaluation_df[evaluation_df['task_target'] == obj]['task_target_idenfication_successful'].sum()
        nb_grip_found_by_objects = {}
        for obj in expected_objects:
            nb_grip_found_by_objects['nb_grip_found_'+obj] = evaluation_df[evaluation_df['task_target'] == obj]['task_grip_idenfication_successful'].sum()
        nb_grip_found_by_grip = {}
        for grip in grips:
            nb_grip_found_by_grip['nb_grip_found_'+grip] = evaluation_df[evaluation_df['task_grip'] == grip]['task_grip_idenfication_successful'].sum()
        
        nb_trials_valid = evaluation_df['is_trial_valid'].sum()
        nb_target_successful = 0
        nb_target_successful_distance = 0
        nb_target_successful_distance_derivative = 0
        nb_target_successful_future_distance = 0
        nb_target_successful_impacts = 0
        nb_target_successful_max_metric = 0
        
        nb_grip_successful = 0
        nb_trials_invalid_and_target_successful = 0
        nb_trials_invalid_and_grip_successful = 0
        
        nb_valid_and_cam_same = 0
        nb_target_successful_and_cam_same = 0
        nb_grip_successful_and_cam_same = 0
        
        nb_valid_and_cam_opposite = 0
        nb_target_successful_and_cam_opposite = 0
        nb_grip_successful_and_cam_opposite = 0
        
        nb_valid_and_executed = 0
        nb_target_successful_and_executed = 0
        nb_grip_successful_and_executed = 0
        
        nb_valid_and_simulated = 0
        nb_target_successful_and_simulated = 0
        nb_grip_successful_and_simulated = 0
        
        total_nb_frames = evaluation_df['trial_nb_frames'].sum()
        total_nb_achievable_frames = evaluation_df['trial_nb_achievable_task_detections'].sum()
        total_frames_target_found = evaluation_df['nb_target_found'].sum()
        
        
        metrics = ['impacts', 'distance', 'future_distance', 'distance_derivative', 'max_metric']

        eval_df_target_found = evaluation_df[evaluation_df['task_target_idenfication_successful']]    
        
        sum_target_found_margin = eval_df_target_found['target_found_margin'].sum()
        mean_target_found_margin = eval_df_target_found['target_found_margin'].mean()
        sd_target_found_margin = eval_df_target_found['target_found_margin'].std()
        sum_target_found_delay = eval_df_target_found['target_found_delay'].sum()
        mean_target_found_delay = eval_df_target_found['target_found_delay'].mean()
        sd_target_found_delay = eval_df_target_found['target_found_delay'].std()   
         
        eval_df_target_found_met =  {}
        sum_target_found_margin_met = {}
        sum_target_found_delay_met = {}
        for metric in metrics:
            eval_df_target_found_met[metric] = evaluation_df[evaluation_df[f'task_target_idenfication_successful_{metric}']]  
            sum_target_found_margin_met[metric] = eval_df_target_found_met[metric][f'target_found_margin_{metric}'].sum()
            sum_target_found_delay_met[metric] = eval_df_target_found_met[metric][f'target_found_delay_{metric}'].sum()
        
        
        eval_df_grip_found = evaluation_df[evaluation_df['task_grip_idenfication_successful']]
        
        sum_grip_found_margin = eval_df_grip_found['grip_found_margin'].sum()
        mean_grip_found_margin = eval_df_grip_found['grip_found_margin'].mean()
        sd_grip_found_margin = eval_df_grip_found['grip_found_margin'].std()
        sum_grip_found_delay = eval_df_grip_found['grip_found_delay'].sum()        
        mean_grip_found_delay = eval_df_grip_found['grip_found_delay'].mean()
        sd_grip_found_delay = eval_df_grip_found['grip_found_delay'].std()
        
        # eval_df_grip_found_met =  {}
        # sum_grip_found_margin_met = {}
        # sum_grip_found_delay_met = {}
        # for metric in metrics:
        #     eval_df_grip_found_met[metric] = evaluation_df[evaluation_df[f'task_grip_idenfication_successful_{metric}']]  
        #     sum_grip_found_margin_met[metric] = eval_df_grip_found_met[metric][f'grip_found_margin_{metric}']
        #     sum_grip_found_delay_met[metric] = eval_df_grip_found_met[metric][f'grip_found_delay_{metric}']
        
        
        mean_trial_duration = evaluation_df['trial_duration'].mean()
        sum_trial_duration = evaluation_df['trial_duration'].sum()
        sd_trial_duration = evaluation_df['trial_duration'].std()
        mean_movement_duration = evaluation_df['movement_duration'].mean()
        sum_movement_duration = evaluation_df['movement_duration'].sum()
        sd_movement_duration = evaluation_df['movement_duration'].std()
        
        mean_time_to_target_rmse = evaluation_df['time_to_target_rmse'].mean()
        
        for index, row in evaluation_df.iterrows():
            if row['is_trial_valid'] and row['task_target_idenfication_successful']:
                nb_target_successful += 1
            if row['is_trial_valid'] and row['task_target_idenfication_successful_distance']:
                nb_target_successful_distance += 1
            if row['is_trial_valid'] and row['task_target_idenfication_successful_distance_derivative']:
                nb_target_successful_distance_derivative += 1
            if row['is_trial_valid'] and row['task_target_idenfication_successful_future_distance']:
                nb_target_successful_future_distance += 1
            if row['is_trial_valid'] and row['task_target_idenfication_successful_impacts']:
                nb_target_successful_impacts += 1
            if row['is_trial_valid'] and row['task_target_idenfication_successful_max_metric']:
                nb_target_successful_max_metric += 1
            if not row['is_trial_valid'] and row['task_target_idenfication_successful']:
                nb_trials_invalid_and_target_successful += 1
            if not row['is_trial_valid'] and row['task_grip_idenfication_successful']:
                nb_trials_invalid_and_grip_successful += 1
            if row['is_trial_valid'] and row['task_grip_idenfication_successful']:
                nb_grip_successful += 1
            if row['is_trial_valid']and row['cam_hand_position']=='same':
                nb_valid_and_cam_same += 1
                if row['task_target_idenfication_successful']:
                    nb_target_successful_and_cam_same += 1
                if row['task_grip_idenfication_successful']:
                    nb_grip_successful_and_cam_same += 1
            if row['is_trial_valid']and row['cam_hand_position']=='opposite':
                nb_valid_and_cam_opposite += 1
                if row['task_target_idenfication_successful']:
                    nb_target_successful_and_cam_opposite += 1
                if row['task_grip_idenfication_successful']:
                    nb_grip_successful_and_cam_opposite += 1
            if row['is_trial_valid']and row['movement_mode'] == 'executed':
                nb_valid_and_executed += 1
                if row['task_target_idenfication_successful']:
                    nb_target_successful_and_executed += 1
                if row['task_grip_idenfication_successful']:
                    nb_grip_successful_and_executed += 1
            if row['is_trial_valid']and row['movement_mode'] == 'simulated':
                nb_valid_and_simulated += 1
                if row['task_target_idenfication_successful']:
                    nb_target_successful_and_simulated += 1
                if row['task_grip_idenfication_successful']:
                    nb_grip_successful_and_simulated += 1
        
        total_nb_direction_correct = evaluation_df['nb_direction_correct'].sum()
        total_nb_impacts_correct = evaluation_df['nb_impacts_correct'].sum()
        total_nb_distance_correct = evaluation_df['nb_distance_correct'].sum()
        total_nb_future_distance_correct = evaluation_df['nb_future_distance_correct'].sum()
        total_nb_distance_derivative_correct = evaluation_df['nb_distance_derivative_correct'].sum()
        total_nb_max_metric_correct = evaluation_df['nb_max_metric_correct'].sum()
        
        total_nb_direction_correct_when_target_not_found = evaluation_df['nb_direction_correct_when_most_probable_wrong'].sum()        
        total_nb_impacts_correct_when_target_not_found = evaluation_df['nb_impacts_correct_when_most_probable_wrong'].sum()
        total_nb_distance_correct_when_target_not_found = evaluation_df['nb_distance_correct_when_most_probable_wrong'].sum()
        total_nb_future_distance_correct_when_target_not_found = evaluation_df['nb_future_distance_correct_when_most_probable_wrong'].sum()
        total_nb_distance_derivative_correct_when_target_not_found = evaluation_df['nb_distance_derivative_correct_when_most_probable_wrong'].sum()
        
        total_nb_direction_correct_when_target_not_found_beginning = evaluation_df['nb_direction_correct_when_most_probable_wrong_beginning'].sum()
        total_nb_impacts_correct_when_target_not_found_beginning = evaluation_df['nb_impacts_correct_when_most_probable_wrong_beginning'].sum()
        total_nb_distance_correct_when_target_not_found_beginning = evaluation_df['nb_distance_correct_when_most_probable_wrong_beginning'].sum()
        total_nb_future_distance_correct_when_target_not_found_beginning = evaluation_df['nb_future_distance_correct_when_most_probable_wrong_beginning'].sum()
        total_nb_distance_derivative_correct_when_target_not_found_beginning = evaluation_df['nb_distance_derivative_correct_when_most_probable_wrong_beginning'].sum()
        
        total_nb_direction_correct_when_target_not_found_end = evaluation_df['nb_direction_correct_when_most_probable_wrong_end'].sum()
        total_nb_impacts_correct_when_target_not_found_end = evaluation_df['nb_impacts_correct_when_most_probable_wrong_end'].sum()
        total_nb_distance_correct_when_target_not_found_end = evaluation_df['nb_distance_correct_when_most_probable_wrong_end'].sum()
        total_nb_future_distance_correct_when_target_not_found_end = evaluation_df['nb_future_distance_correct_when_most_probable_wrong_end'].sum()
        total_nb_distance_derivative_correct_when_target_not_found_end = evaluation_df['nb_distance_derivative_correct_when_most_probable_wrong_end'].sum()
        
        total_nb_target_not_found_but_individual_metrics_correct = evaluation_df['nb_target_not_found_but_individual_metrics_correct'].sum()
        
        if nb_trials_valid == 0:
            ratio_target_successful = 0
            ratio_grip_successful = 0
        else:
            ratio_target_successful = nb_target_successful/nb_trials_valid
            ratio_grip_successful = nb_grip_successful/nb_trials_valid
            
        if nb_valid_and_cam_same == 0:
            ratio_target_successful_and_cam_same = 0
            ratio_grip_successful_and_cam_same = 0
        else:
            ratio_target_successful_and_cam_same = nb_target_successful_and_cam_same/nb_valid_and_cam_same
            ratio_grip_successful_and_cam_same = nb_grip_successful_and_cam_same/nb_valid_and_cam_same
        
        if nb_valid_and_cam_opposite == 0:
            ratio_target_successful_and_cam_opposite = 0
            ratio_grip_successful_and_cam_opposite = 0
        else:
            ratio_target_successful_and_cam_opposite = nb_target_successful_and_cam_opposite/nb_valid_and_cam_opposite
            ratio_grip_successful_and_cam_opposite = nb_grip_successful_and_cam_opposite/nb_valid_and_cam_opposite
        
        summary = { 'nb_trials': nb_trials,
                   'nb_cam_same': nb_cam_same,
                   'nb_cam_opposite': nb_cam_opposite,
                   'nb_executed': nb_executed,
                     'nb_simulated': nb_simulated,
                   
                   
                   'nb_trials_valid': nb_trials_valid, 
                   'nb_valid_and_cam_same': nb_valid_and_cam_same,
                   'nb_valid_and_cam_opposite': nb_valid_and_cam_opposite,
                   'nb_valid_and_executed': nb_valid_and_executed,
                   'nb_valid_and_simulated': nb_valid_and_simulated,
                   
                   'nb_target_successful': nb_target_successful, 
                   'nb_target_successful_distance': nb_target_successful_distance,
                     'nb_target_successful_distance_derivative': nb_target_successful_distance_derivative,
                        'nb_target_successful_future_distance': nb_target_successful_future_distance,
                        'nb_target_successful_impacts': nb_target_successful_impacts,
                        'nb_target_successful_max_metric': nb_target_successful_max_metric,
                        
                   'nb_target_successful_and_cam_same': nb_target_successful_and_cam_same,
                   'nb_target_successful_and_cam_opposite': nb_target_successful_and_cam_opposite,
                   'nb_target_successful_and_executed': nb_target_successful_and_executed,
                   'nb_target_successful_and_simulated': nb_target_successful_and_simulated,
                   
                   'total_nb_frames': total_nb_frames,
                   'total_nb_achievable_frames': total_nb_achievable_frames,
                   'total_nb_frames_target_found': total_frames_target_found,
                   'total_nb_frames_target_not_found': total_nb_achievable_frames-total_frames_target_found,
                    'nb_target_not_found_but_individual_metrics_correct': total_nb_target_not_found_but_individual_metrics_correct,
                   
                    'nb_direction_correct': total_nb_direction_correct,
                   'nb_impacts_correct': total_nb_impacts_correct,
                    'nb_distance_correct': total_nb_distance_correct,
                    'nb_future_distance_correct': total_nb_future_distance_correct,
                    'nb_distance_derivative_correct': total_nb_distance_derivative_correct,
                    'nb_max_metric_correct': total_nb_max_metric_correct,
                    
                    
                    'nb_direction_correct_when_target_not_found': total_nb_direction_correct_when_target_not_found,
                    'nb_impacts_correct_when_target_not_found': total_nb_impacts_correct_when_target_not_found,
                    'nb_distance_correct_when_target_not_found': total_nb_distance_correct_when_target_not_found,
                    'nb_future_distance_correct_when_target_not_found': total_nb_future_distance_correct_when_target_not_found,
                    'nb_distance_derivative_correct_when_target_not_found': total_nb_distance_derivative_correct_when_target_not_found,
                    
                    'nb_direction_correct_when_target_not_found_beginning': total_nb_direction_correct_when_target_not_found_beginning,
                    'nb_impacts_correct_when_target_not_found_beginning': total_nb_impacts_correct_when_target_not_found_beginning,
                    'nb_distance_correct_when_target_not_found_beginning': total_nb_distance_correct_when_target_not_found_beginning,
                    'nb_future_distance_correct_when_target_not_found_beginning': total_nb_future_distance_correct_when_target_not_found_beginning,
                    'nb_distance_derivative_correct_when_target_not_found_beginning': total_nb_distance_derivative_correct_when_target_not_found_beginning,
                    
                    'nb_direction_correct_when_target_not_found_end': total_nb_direction_correct_when_target_not_found_end,
                    'nb_impacts_correct_when_target_not_found_end': total_nb_impacts_correct_when_target_not_found_end,
                    'nb_distance_correct_when_target_not_found_end': total_nb_distance_correct_when_target_not_found_end,
                    'nb_future_distance_correct_when_target_not_found_end': total_nb_future_distance_correct_when_target_not_found_end,
                    'nb_distance_derivative_correct_when_target_not_found_end': total_nb_distance_derivative_correct_when_target_not_found_end,
                    
                    'ratio_direction_correct': total_nb_direction_correct/total_nb_achievable_frames,
                    'ratio_impacts_correct': total_nb_impacts_correct/total_nb_achievable_frames,
                    'ratio_distance_correct': total_nb_distance_correct/total_nb_achievable_frames,
                    'ratio_distance_derivative_correct': total_nb_distance_derivative_correct/total_nb_achievable_frames,
                    
                    'ratio_direction_correct_when_target_not_found': total_nb_direction_correct_when_target_not_found/(total_nb_achievable_frames-total_frames_target_found),
                    'ratio_impacts_correct_when_target_not_found': total_nb_impacts_correct_when_target_not_found/(total_nb_achievable_frames-total_frames_target_found),
                    'ratio_distance_correct_when_target_not_found': total_nb_distance_correct_when_target_not_found/(total_nb_achievable_frames-total_frames_target_found),
                    'ratio_distance_derivative_correct_when_target_not_found': total_nb_distance_derivative_correct_when_target_not_found/(total_nb_achievable_frames-total_frames_target_found),
                    
                    
                   'ratio_target_successful': ratio_target_successful, 
                   'ratio_target_successful_and_cam_same': ratio_target_successful_and_cam_same,
                   'ratio_target_successful_and_cam_opposite': ratio_target_successful_and_cam_opposite,
                   
                   'nb_grip_successful': nb_grip_successful, 
                   'nb_grip_successful_and_cam_same': nb_grip_successful_and_cam_same,
                   'nb_grip_successful_and_cam_opposite': nb_grip_successful_and_cam_opposite,
                   'nb_grip_successful_and_executed': nb_grip_successful_and_executed,
                   'nb_grip_successful_and_simulated': nb_grip_successful_and_simulated,
                   
                   'ratio_grip_successful': ratio_grip_successful,
                   'ratio_grip_successful_and_cam_same': ratio_grip_successful_and_cam_same,
                   'ratio_grip_successful_and_cam_opposite': ratio_grip_successful_and_cam_opposite,
                   'ratio_grip_successful_and_executed': nb_grip_successful_and_executed/nb_valid_and_executed,
                   'ratio_grip_successful_and_simulated': nb_grip_successful_and_simulated/nb_valid_and_simulated,
                   
                   'nb_trials_invalid_and_target_successful': nb_trials_invalid_and_target_successful,
                    'nb_trials_invalid_and_grip_successful': nb_trials_invalid_and_grip_successful,
                    
                    'sum_target_found_margin': sum_target_found_margin,
                    'mean_target_found_margin': mean_target_found_margin,
                    'sd_target_found_margin': sd_target_found_margin,
                    'sum_grip_found_margin': sum_grip_found_margin,
                    'mean_grip_found_margin': mean_grip_found_margin,
                    'sd_grip_found_margin': sd_grip_found_margin,
                    
                    'sum_target_found_delay': sum_target_found_delay,
                    'mean_target_found_delay': mean_target_found_delay,
                    'sd_target_found_delay': sd_target_found_delay,
                    'sum_grip_found_delay': sum_grip_found_delay,
                    'mean_grip_found_delay': mean_grip_found_delay,
                    'sd_grip_found_delay': sd_grip_found_delay,
                    
                    'sum_target_found_margin_impacts': sum_target_found_margin_met['impacts'],
                    'sum_target_found_margin_distance': sum_target_found_margin_met['distance'],
                    'sum_target_found_margin_future_distance': sum_target_found_margin_met['future_distance'],
                    'sum_target_found_margin_distance_derivative': sum_target_found_margin_met['distance_derivative'],
                    'sum_target_found_margin_max_metric': sum_target_found_margin_met['max_metric'],
                    
                    'sum_target_found_delay_impacts': sum_target_found_delay_met['impacts'],
                    'sum_target_found_delay_distance': sum_target_found_delay_met['distance'],
                    'sum_target_found_delay_future_distance': sum_target_found_delay_met['future_distance'],
                    'sum_target_found_delay_distance_derivative': sum_target_found_delay_met['distance_derivative'],
                    'sum_target_found_delay_max_metric': sum_target_found_delay_met['max_metric'],
                    
                    # 'sum_grip_found_margin_impacts': sum_grip_found_margin_met['impacts'],
                    # 'sum_grip_found_margin_distance': sum_grip_found_margin_met['distance'],    
                    # 'sum_grip_found_margin_future_distance': sum_grip_found_margin_met['future_distance'],
                    # 'sum_grip_found_margin_distance_derivative': sum_grip_found_margin_met['distance_derivative'],
                    # 'sum_grip_found_margin_max_metric': sum_grip_found_margin_met['max_metric'],
                    
                    # 'sum_grip_found_delay_impacts': sum_grip_found_delay_met['impacts'],
                    # 'sum_grip_found_delay_distance': sum_grip_found_delay_met['distance'],
                    # 'sum_grip_found_delay_future_distance': sum_grip_found_delay_met['future_distance'],
                    # 'sum_grip_found_delay_distance_derivative': sum_grip_found_delay_met['distance_derivative'],
                    # 'sum_grip_found_delay_max_metric': sum_grip_found_delay_met['max_metric'],
                    
                    
                    'sum_trial_duration': sum_trial_duration,
                    'mean_trial_duration': mean_trial_duration,
                    'sd_trial_duration': sd_trial_duration,
                    'sum_movement_duration': sum_movement_duration,
                    'mean_movement_duration': mean_movement_duration,
                    'sd_movement_duration': sd_movement_duration,
                    
                   'mean_time_to_target_rmse': mean_time_to_target_rmse,
                   **nb_trials_valid_by_objects,
                     **nb_trials_valid_by_grip,
                        **nb_target_found_by_objects,
                        **nb_grip_found_by_objects,
                        **nb_grip_found_by_grip
                   }
        summary_df = pd.DataFrame.from_dict(summary, orient='index')
        summary_df.to_csv(os.path.join(self.evaluation_path, f"{self.pseudo}_{device_ID}_summary.csv"), index=True)
            
    
    def get_number_of_trials(self):
        return len(self.available_trials)
    
    def check_processed(self):
        if os.path.exists(self.data_csv_path):
            self.processed = True
        else:
            self.processed = False
            
    def is_processed(self):
        return self.processed
    
    def is_all_data_available(self):
        return self.all_data_available
    
    def is_folder_available(self):
        return self.path is not None
    
    def is_combinations_available(self):
        return self.combinations_data is not None
    
    def is_all_trial_folders_available(self):
        return len(self.missing_trial_folders) == 0

    def set_progress_display(self, progress_window, progress_display):
        self.progress_window = progress_window
        self.progress_display = progress_display
        progress_display.reset(len(self.available_trials), "trials pre-processed", f"Processing participant {self.pseudo}")
        
    def set_instructions(self, session_instructions):
        #extract columns 'Label' and language from session_instructions
        self.instructions = session_instructions[['Label', self.language]]
        #change self.language column name to 'Instructions'
        self.instructions.rename(columns={self.language: 'Instructions'}, inplace=True)
        for trial in self.missing_trials:
            trial.set_instructions(self.instructions)
                
    def save_combinations(self):
        self.combinations_data.to_csv(self.combinations_path, index=False)
        print(f"Combinations written to '{self.combinations_path}'")
        
    
    def save_experimental_parameters(self):
        csv_path = os.path.join(self.path, f'{self.pseudo}{Session._EXPERIMENTAL_PARAMETERS_SUFFIX}')
        #check if the file already exists
        if os.path.exists(csv_path):
            overwrite = messagebox.askyesno("File already exists", f"File {csv_path} already exists. Do you want to overwrite it?")
            if overwrite:
                #copy the file to a backup, adding a timestamp
                csv_backup_path = os.path.join(self.path, f"bckp_experimental_parameters_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv")
                os.rename(csv_path, csv_backup_path)
                print('Parameters file backuped to {csv_backup_path}')
            else:
                return
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            params=[]
            for param_type, param_list in self.experimental_parameters.items():
                params.append([param_type]+param_list)
            writer.writerows(params,)
        print(f"Parameters written to '{csv_path}'")
        
    def save_recording_parameters(self):
        csv_path = os.path.join(self.path, f'{self.pseudo}{Session._RECORDING_PARAMETERS_SUFFIX}')
        #check if the file already exists
        if os.path.exists(csv_path):
            overwrite = messagebox.askyesno("File already exists", f"File {csv_path} already exists. Do you want to overwrite it?")
            if overwrite:
                #copy the file to a backup, adding a timestamp
                csv_backup_path = os.path.join(self.path, f"bckp_recording_parameters_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv")
                os.rename(csv_path, csv_backup_path)
                print('Parameters file backuped to {csv_backup_path}')
            else:
                return
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            params=[]
            for param_type, param_list in self.recording_parameters.items():
                params.append([param_type]+param_list)
            writer.writerows(params,)
        print(f"Parameters written to '{csv_path}'")
        
    def close(self):
        self.stop_experiment()
        
class Trial:
    def __init__(self, label, participant_path, combination:pd.DataFrame=None, participant_pre_processing_path = None, participant_replay_path = None, participant_analysis_path=None) -> None:
        self.label = label
        self.combination = combination
        self.participant_path = participant_path
        self.hand_data = None
        self.object_data = None
        self.path = os.path.join(self.participant_path, self.label)
        self.pre_processing_path = os.path.join(participant_pre_processing_path, self.label)
        self.replay_path = os.path.join(participant_replay_path, self.label)
        # self.analysis_path = os.path.join(participant_analysis_path, self.label)
        self.analysis_path = participant_analysis_path
        self.duration = None
        self.meta_data = None
        
        self.ongoing = False
        self.obj_ind = 1
        self.hand_ind = 2
        self.grip_ind = 3
        self.movement_type_ind = 4
        self.combination_header = ["Trial Number", "Objects", "Hands", "Grips", "Movement Types"]
        
        self.save_overlayed_video = False
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    def set_instructions(self, instructions):
        #transform the instructions dataframe into a dictionary
        self.instructions = instructions.set_index('Label').to_dict()['Instructions']
        
    def get_combination(self):
        return self.combination
    
    def was_pre_processed(self, device_ID = None):
        if not os.path.exists(self.pre_processing_path):
            pre_processed = False
            print('Trial {} not pre-processed'.format(self.label))
            return pre_processed
        pre_processed = True
        file_suffixes =  ['depth_map_movement.gzip', 
                          'timestamps_movement.gzip', 
                          'video_movement.avi',
                          'depth_map_contact.gzip', 
                          'timestamps_contact.gzip',
                          'video_contact.avi'
                        #   'depth_map_return.gzip',
                        #   'timestamps_return.gzip',
                        #   'video_return.avi',
                        #   'depth_map_stand.gzip',
                        #   'timestamps_stand.gzip',
                        #   'video_stand.avi'
                          ]

        for suffix in file_suffixes:
            #count number of files with suffix in the trial folder
            if device_ID is not None:
                file_count = len([f for f in os.listdir(self.pre_processing_path) if f.endswith(suffix) and device_ID in f])
                nmin = 1
            else:
                file_count = len([f for f in os.listdir(self.pre_processing_path) if f.endswith(suffix)])
                nmin = 2
            if file_count <nmin:
                pre_processed = False
                if device_ID is not None:
                    print(f"Trial '{self.label}' not pre-processed: missing file with suffix '{suffix}' and device ID '{device_ID}'")
                else:
                    print(f"Trial '{self.label}' not pre-processed: missing file with suffix '{suffix}'")
                break
        if pre_processed:
            if device_ID is not None:
                print('CHECKED : Trial {} has already been pre-processed for device {}'.format(self.label, device_ID))
            else:
                print('CHECKED : Trial {} has already been pre-processed'.format(self.label))
        else:
            if device_ID is not None:
                print('CHECKED : Trial {} has never been pre-processed for device {}'.format(self.label, device_ID))
            else: 
                print('CHECKED : Trial {} has never been pre-processed'.format(self.label))
        
        return pre_processed
    
    def was_replayed(self, device_ID = None):
        if not os.path.exists(self.replay_path):
            replayed = False
            return replayed
        replayed = True
        files_suffixes = ['hand_traj.csv', 
                          'main.csv', 
                          'replay_data.pkl',
                          'monitoring.csv',]
        for suffix in files_suffixes:
            if device_ID is not None:
                file_count = len([f for f in os.listdir(self.replay_path) if f.endswith(suffix) and device_ID in f])
                nmin = 1
            else:
                file_count = len([f for f in os.listdir(self.replay_path) if f.endswith(suffix)])
                nmin = 2
            if file_count <nmin:
                replayed = False
                print(f"Trial '{self.label}' not replayed: missing file with suffix '{suffix}'")
                break
        if device_ID is not None:
            file_count = len([f for f in os.listdir(self.replay_path) if f.endswith('traj.csv')and 'obj' in f and device_ID in f])
            nmin = 1
        else:
            file_count = len([f for f in os.listdir(self.replay_path) if f.endswith('traj.csv')and 'obj' in f])
            nmin = 2
        if file_count <nmin:
            replayed = False
            print(f"Trial '{self.label}' not replayed: at least one object traj file should be present")
        if replayed:
            if device_ID is not None :
                print('CHECKED : {} has already been replayed for device {}'.format(self.label, device_ID))
            else:
                print('CHECKED : {} has already been replayed'.format(self.label))
        else:
            if device_ID is not None :
                print('CHECKED : {} has never been replayed for device {}'.format(self.label, device_ID))
            else:
                print('CHECKED : {} has never been replayed'.format(self.label))
            
        return replayed
    
    def was_analysed(self, device_ID = None):
        if not os.path.exists(self.analysis_path):
            analysed = False
            return analysed
        analysed = True
        file_suffix = 'target_data.csv'
        if device_ID is not None:
            file_count = len([f for f in os.listdir(self.analysis_path) if f.endswith(file_suffix) and device_ID in f and self.label in f])
            nmin = 1
        else:
            file_count = len([f for f in os.listdir(self.analysis_path) if f.endswith(file_suffix) and self.label in f])
            nmin = 2
        if file_count <nmin:
            analysed = False
            if device_ID is not None:
                print(f"Trial '{self.label}' not analysed: missing file with suffix '{file_suffix}' and device ID '{device_ID}'")
            else:
                print(f"Trial '{self.label}' not analysed: missing file with suffix '{file_suffix}'")
        return analysed
    
    def pre_process(self, experiment_pre_processor):
        if not os.path.exists(self.path):
            print(f"Trial {self.label} raw data folder not found. This trial cannot be pre-processed and will be skipped.")
            return False, False, None
        print(f"Pre-processing trial {self.label}")
        if not os.path.exists(self.pre_processing_path):
            os.mkdir(self.pre_processing_path)
        self.combi_ok, self.face_ok, duration = experiment_pre_processor.process_trial(self.path, self.combination, self.pre_processing_path)
        return self.combi_ok, self.face_ok, duration
    
    def replay(self, experiment_replayer, sequence = 'movement'):
        if not self.was_pre_processed():
            print(f'Trial {self.label} not pre-processed. This trial cannot be replayed and will be skipped.')
            return {}
        print(f"Replaying trial {self.label}")
        if not os.path.exists(self.replay_path):
            os.mkdir(self.replay_path)
        device_id = experiment_replayer.get_device_id()
        # print('device_id', device_id)
        # print('folder', self.pre_processing_path)
        #get the .gzip file with device_id in the name
        depth_file_list = [f for f in os.listdir(self.pre_processing_path) if device_id in f and f.endswith(".gzip") and 'depth_map' in f and sequence in f]
        # print('depth_file_list', depth_file_list)
        depth_file = depth_file_list[0]
        #extract data from the first file into a dataframe
        timestamps_and_depth = pd.read_pickle(os.path.join(self.pre_processing_path, depth_file), compression='gzip')
        #get the video file with device_id in the name
        video = [f for f in os.listdir(self.pre_processing_path) if device_id in f and f.endswith(".avi") and sequence in f][0]
        #merge the two dataframes into a single dataframe
        replay = timestamps_and_depth.to_dict(orient='list')
        replay['Video'] = os.path.join(self.pre_processing_path, video)
        
        #get the current pandas timestamp
        now = pd.Timestamp.now()
        
        #get task hand and object from the combination dataframe
        task_object = self.combination[self.combination_header[self.obj_ind]]
        task_hand = self.combination[self.combination_header[self.hand_ind]]
        
        #replay the experiment trial, and extract hands_data and objects_data
        self.hands_data, self.objects_data, self.replay_monitoring, self.saved_imgs, self.replay_data_dict = experiment_replayer.replay(replay, task_hand, task_object)
        
        
        # compute the duration of replaying the trial
        replay_duration = (pd.Timestamp.now() - now).total_seconds()
        
        #compute the duration of the trial
        # get first and last timestamps and compute the duration of the trial
        first_timestamp = timestamps_and_depth['Timestamps'].iloc[0]
        last_timestamp = timestamps_and_depth['Timestamps'].iloc[-1]
        self.duration = last_timestamp - first_timestamp
        self.meta_data = {'Trial_duration': [self.duration], 'Trial_data_extration_duration': [replay_duration]}
        
        timestamps_only = pd.read_pickle(os.path.join(self.pre_processing_path, f"{self.label}_cam_{device_id}_timestamps_{sequence}.gzip"), compression='gzip')
        self.main_data = timestamps_only
        
        # merge the hands_data into the main_data
        hand_keys = sc.GraspingHand.MAIN_DATA_KEYS
        for hand_id, hand_data in self.hands_data.items():
            hand_summary = pd.DataFrame()
            hand_summary['Timestamps'] = hand_data['Timestamps']
            for key in hand_keys:
                if key != 'Timestamps':
                    hand_summary[hand_id + '_' + key] = hand_data[key]
            # add the hand_summary to the main_data starting at the row corresponding to the first timestamp
            self.main_data = pd.merge(self.main_data, hand_summary, on='Timestamps', how='left')
        
        
        # merge the objects_data into the main_data
        object_keys = sc.RigidObject.MAIN_DATA_KEYS
        for object_id, object_data in self.objects_data.items():            
            object_summary = pd.DataFrame()
            object_summary['Timestamps'] = object_data['Timestamps']
            for key in object_keys:
                if key != 'Timestamps':
                    object_summary[object_id + '_' + key] = object_data[key]
            
            # add the object_summary to the main_data starting at the row corresponding to the first timestamp
            self.main_data = pd.merge(self.main_data, object_summary, on='Timestamps', how='left')
        
        # save the replay data to csv files
        self.save_replay_data(device_id)
        
        return self.meta_data

    def analyse(self, experiment_analyser, evaluate=False):
        
        if not self.was_replayed():
            print(f'Trial {self.label} not replayed. This trial cannot be analysed and will be skipped.')
            return
        
        # print(f"Analysing trial {self.label}")
        # if not os.path.exists(self.analysis_path):
        #     os.mkdir(self.analysis_path)
            
        device_id = experiment_analyser.get_device_id()
        
        
        #get task hand and object from the combination dataframe
        task_object = self.combination[self.combination_header[self.obj_ind]]
        task_hand = self.combination[self.combination_header[self.hand_ind]]
        task_grip = self.combination[self.combination_header[self.grip_ind]]
        
        replay_data_file = [f for f in os.listdir(self.replay_path) if f.endswith('replay_data.pkl') and device_id in f][0]
        video_name = [f for f in os.listdir(self.replay_path) if device_id in f and f.endswith("overlayed_video.avi") ][0]
        monitoring_file = [f for f in os.listdir(self.replay_path) if f.endswith('monitoring.csv') and device_id in f][0]
        video_path = os.path.join(self.replay_path, video_name)
        # print(f'video_path: {video_path}')

        with open(os.path.join(self.replay_path, replay_data_file), 'rb') as f:
            replay_data = pickle.load(f)
            
        print(f"replay_data: {replay_data}")
        if 'timestamps' in replay_data.keys():
            replay_timestamps = replay_data['timestamps']
        elif 'timestamp' in replay_data.keys():
            replay_timestamps = replay_data['timestamp']
            
        replay_hands = replay_data['hands']
        replay_objects = replay_data['objects']
        save_scene_path = os.path.join(self.analysis_path, f"{self.label}_cam_{device_id}_scene")
        target_data = experiment_analyser.analyse(task_hand, task_object, task_grip, replay_timestamps, replay_hands, replay_objects, video_path, save_scene_path=save_scene_path)
        
        monitoring_data = pd.read_csv(os.path.join(self.replay_path, monitoring_file))
        # drop the 'timestamp' column if it exists
        if 'timestamp' in target_data.columns:
            target_data.drop(columns=['timestamp'], inplace=True)
        # target_data.drop(columns=['timestamp'], inplace=True)
        # concatenate the monitoring data to the target data
        target_data = pd.concat([monitoring_data, target_data], axis=1)

        print(f"target_data: {target_data}")
        
        target_data.to_csv(os.path.join(self.analysis_path, f"{self.label}_cam_{device_id}_target_data.csv"), index=False)
        if evaluate:
            evaluation = self.evaluate(target_data, device_id)
        else:
            evaluation = None
        return evaluation
        
    def evaluate(self, target_data= None, device_ID = None):
        if target_data is None and device_ID is None:
            return None
        
        if target_data is None:
            if os.path.exists(os.path.join(self.analysis_path, f"{self.label}_cam_{device_ID}_target_data.csv")):
                target_data = pd.read_csv(os.path.join(self.analysis_path, f"{self.label}_cam_{device_ID}_target_data.csv"))
            else:
                return None
        task_hand = self.combination[self.combination_header[self.hand_ind]]
        task_object = self.combination[self.combination_header[self.obj_ind]]
        task_grip = self.combination[self.combination_header[self.grip_ind]]
        mode = self.combination[self.combination_header[self.movement_type_ind]]
        return Trial.evaluate_target_data(target_data, device_ID=device_ID, hand=task_hand, target=task_object, mode=mode, grip= task_grip)  

    def evaluate_target_data_26_04(target_data, device_ID = None, hand = None, target = None):     # 26_04         
        
        print(f"evaluate target_data: {target_data}")
        # print df columns headers
        print(f"columns: {target_data.columns}")
        ## TRIAL METADATA
        trial_duration = target_data['timestamp'].iloc[-1] - target_data['timestamp'].iloc[0]
        trial_nb_frames = len(target_data)
        if device_ID is not None:
            if device_ID == '1944301011EA1F1300':
                cam_position = 'right'
            elif device_ID == '19443010910F481300':
                cam_position = 'left'
        if hand is not None:
            if hand == cam_position:
                cam_hand_position = 'same'
            else:
                cam_hand_position = 'opposite'                
                
        
        ## HAND DETECTION
        
        #find the first True value in the 'task_hand_found' column
        idx_first_hand_found = target_data['task_hand_found'].idxmax()
        first_hand_found_timestamp = target_data.loc[idx_first_hand_found, 'timestamp']
        nb_hand_found = target_data['task_hand_found'].sum()
        ratio_hand_found = nb_hand_found / trial_nb_frames
        
        #check if the hand was continuously estimated
        is_hand_estimation_continuous = target_data.loc[idx_first_hand_found:,'task_hand_found'].all()
        
        ## OBJECT DETECTION
        
        #find the first True value in the 'task_object_found' column
        first_object_found = target_data['task_object_found'].idxmax()
        first_object_found_timestamp = target_data.loc[first_object_found, 'timestamp']
        nb_object_found = target_data['task_object_found'].sum()
        ratio_object_found = target_data['task_object_found'].sum() / trial_nb_frames
        
        #check the column 'task_object_estimation_continuous' to see if the object was continuously estimated
        is_object_estimation_continuous = target_data.loc[first_object_found:,'task_object_found'].all()
        
        ## TRIAL VALIDATION
        
        achievable_index = max(idx_first_hand_found, first_object_found)
        achievable_timestamp = target_data.loc[achievable_index, 'timestamp']
        achievable_time = trial_duration - achievable_timestamp 
        nb_achievable_task_detections = trial_nb_frames - achievable_index
        ratio_achievable_task_detections = nb_achievable_task_detections / trial_nb_frames

        # check hand detection
        max_nb_consecutive_failed_hand_detections = 3
        too_many_consecutive_failed_hand_detections = False
        nb_consecutive_failed_hand_detections =0
        for index, row in target_data.iterrows():
            if row['task_hand_found'] == True :
                nb_consecutive_failed_hand_detections = 0
            else:
                nb_consecutive_failed_hand_detections += 1
            if nb_consecutive_failed_hand_detections > max_nb_consecutive_failed_hand_detections:
                too_many_consecutive_failed_hand_detections = True
                break
        
        # check object detection
        max_nb_consecutive_failed_object_detections = 3
        too_many_consecutive_failed_object_detections = False
        nb_consecutive_failed_object_detections = 0
        for index, row in target_data.iterrows():
            if row['task_object_found'] == True :
                nb_consecutive_failed_object_detections = 0
            else:
                nb_consecutive_failed_object_detections += 1
            if nb_consecutive_failed_object_detections > max_nb_consecutive_failed_object_detections:
                too_many_consecutive_failed_object_detections = True
                break
        
        # define the minimum ratio of hand and object found to consider the trial valid
        min_ratio_hand_found = 0.5
        min_ratio_object_found = 0.5
        min_ratio_achievable_task_detections = 0.4
        min_achievable_time = 0.5
        
        hand_found_ok = ratio_hand_found > min_ratio_hand_found
        object_found_ok = ratio_object_found > min_ratio_object_found
        achievable_task_detections_ok = ratio_achievable_task_detections > min_ratio_achievable_task_detections
        achievable_time_ok = achievable_time > min_achievable_time
        
        # is_trial_valid =  hand_found_ok and object_found_ok and achievable_task_detections_ok and achievable_time_ok and not too_many_consecutive_failed_hand_detections and not too_many_consecutive_failed_object_detections
        is_trial_valid = achievable_time_ok and not too_many_consecutive_failed_hand_detections and not too_many_consecutive_failed_object_detections
        not_valid_reasons = []
        if not is_trial_valid:
            if ratio_hand_found <= 0.7:
                not_valid_reasons.append('hand not found enough')
            if ratio_object_found <= 0.9:
                not_valid_reasons.append('object not found enough')
            if too_many_consecutive_failed_hand_detections:
                not_valid_reasons.append('too many consecutive failed hand detections')
            if too_many_consecutive_failed_object_detections:
                not_valid_reasons.append('too many consecutive failed object detections') 
        
        # create a new column 'task_estimation_achievable' with False before achievable_index and True after
        target_data['task_estimation_achievable'] = False
        target_data.loc[achievable_index:, 'task_estimation_achievable'] = True
              
        ## TARGET IDENTIFICATION
        
        #find the first True value in the 'task_object_found' column
        idx_first_target_found = target_data['task_target_found'].idxmax() 
        first_target_found_timestamp = target_data.loc[idx_first_target_found, 'timestamp']- achievable_timestamp
        
        #count the number of True values in the 'task_object_found' column
        nb_target_found = target_data['task_target_found'].sum()
        
        # compute the ratio of target found
        target_found_ratio = nb_target_found / nb_achievable_task_detections
        
        # count the number of False values in the 'task_object_found' column after the first True value
        after_first_found = target_data.loc[idx_first_target_found:, 'task_target_found']
        ratio_target_switch = (len(after_first_found) - after_first_found.sum())/len(after_first_found)
        task_target_idenfication_successful = ratio_target_switch < 0.3
        
         
        # for each metric, check if colmns target_from_impacts	target_from_distance, target_from_distance_derivative and target_from_direction are have same value as target
        
        metrics = ['impacts', 'distance', 'distance_derivative', 'direction']
        for metric in metrics:
            # test if 'target_from_{metric}' exists in the columns
            if f'target_from_{metric}' not in target_data.columns:
                target_data[f'target_from_{metric}_correct'] = False
            else:
                target_data[f'target_from_{metric}_correct'] = target_data[f'target_from_{metric}'].eq(target)
        
        nb_metric_correct = {}
        for metric in metrics:
            nb_metric_correct[metric] = target_data[f'target_from_{metric}_correct'].sum()
        
        most_trustworthy_metric = max(nb_metric_correct, key=nb_metric_correct.get)
        
        # create a new column 'task_target_not_found_but_individual_metrics_correct' with True if the target was not found but at least one of the individual metrics is correct
        target_data['task_target_not_found_but_individual_metrics_correct'] = False
        target_data['task_target_not_found_but_metrics_correct'] = ''
        for index, row in target_data.iterrows():
            if row['task_target_found'] == False:
                correct_metrics = []
                for metric in metrics:
                    if row[f'target_from_{metric}_correct']:
                        correct_metrics.append(metric)
                        target_data.loc[index, 'task_target_not_found_but_individual_metrics_correct'] = True
                target_data.loc[index, 'task_target_not_found_but_metrics_correct'] = ', '.join(correct_metrics)
                        
        # count the number of True values in the 'task_target_not_found_but_individual_metrics_correct' column
        nb_target_not_found_but_individual_metrics_correct = target_data['task_target_not_found_but_individual_metrics_correct'].sum()
        
        # compute the ratio of target not found but individual metrics correct
        ratio_target_not_found_but_individual_metrics_correct = nb_target_not_found_but_individual_metrics_correct / nb_achievable_task_detections
        # count the number of True values in the 'task
        
        ## GRIP IDENTIFICATION
        
        #find the first True value in the 'task_grip_found' column
        first_grip_found = target_data['task_grip_found'].idxmax() 
        first_grip_found_timestamp = target_data.loc[first_grip_found, 'timestamp']-achievable_timestamp
        
        #count the number of True values in the 'task_grip_found' column
        nb_grips_found = target_data['task_grip_found'].sum()
        
        #compute the ratio of grip found
        grip_found_ratio = nb_grips_found / nb_achievable_task_detections
        
        # count the number of False values in the 'task_grip_found' column after the first True value
        after_first_found = target_data.loc[first_grip_found:, 'task_grip_found']   
        ratio_grip_switch = (len(after_first_found) - after_first_found.sum())/len(after_first_found)
        task_grip_idenfication_successful = ratio_grip_switch < 0.3
        
        ## TIME TO TARGET ESTIMATION
        # add a new column 'time_to_target' with the time to target, compute the difference between the timestamp and the timestamp of the last row
        target_data['time_to_target'] = target_data['timestamp'] - target_data['timestamp'].iloc[-1]
        
        # add a new column 'time_to_target_error' with the difference between the time to target estimation and the actual time to target
        target_data['time_to_target_error'] = target_data['time_to_target'] - target_data['estimated_target_time_to_impact']
        
        # compute the rmse of the time to target estimation
        time_to_target_rmse = np.sqrt(np.mean(target_data['time_to_target_error']**2))
        
        evaluation_data = {'cam_hand_position': cam_hand_position,
                           'trial_nb_frames': trial_nb_frames,
                           'trial_nb_achievable_task_detections': nb_achievable_task_detections,
                           
                           'idx_first_hand_found': idx_first_hand_found,
                           'nb_hand_found': nb_hand_found,
                           'ratio_hand_found': ratio_hand_found,
                           'is_hand_estimation_continuous': is_hand_estimation_continuous,
                           
                           'idx_first_object_found': first_object_found,
                           'nb_object_found': nb_object_found,
                           'ratio_object_found': ratio_object_found,
                           'is_object_estimation_continuous': is_object_estimation_continuous,
                           
                           'trial_duration': trial_duration,
                           'first_hand_found': first_hand_found_timestamp,
                           'first_object_found': first_object_found_timestamp,                           
                           'first_target_found': first_target_found_timestamp, 
                           'first_grip_found': first_grip_found_timestamp, 
                           
                           'is_trial_valid' : is_trial_valid, 
                           'not_valid_reasons': not_valid_reasons,
                           
                           'nb_target_found': nb_target_found, 
                            'nb_impacts_correct': nb_metric_correct['impacts'],
                            'nb_distance_correct': nb_metric_correct['distance'],
                            'nb_distance_derivative_correct': nb_metric_correct['distance_derivative'],
                            'nb_direction_correct': nb_metric_correct['direction'],
                            'most_trustworthy_metric': most_trustworthy_metric,
                            'ratio_target_not_found_but_individual_metrics_correct': ratio_target_not_found_but_individual_metrics_correct,
                            'nb_target_not_found_but_individual_metrics_correct': nb_target_not_found_but_individual_metrics_correct,
                            
                           'target_found_ratio': target_found_ratio, 
                           'ratio_target_switch': ratio_target_switch, 
                           'task_target_idenfication_successful': task_target_idenfication_successful,
                           
                           'nb_grips_found': nb_grips_found,
                           'grip_found_ratio': grip_found_ratio,
                           'ratio_grip_switch': ratio_grip_switch,
                           'task_grip_idenfication_successful': task_grip_idenfication_successful,
                           'time_to_target_rmse': time_to_target_rmse}
        
        return evaluation_data
    
    def evaluate_target_data(target_data, device_ID = None, hand = None, target = None, mode = None, grip= None):     # 16_05_save
        
        print(f"evaluate target_data: {target_data}")
        ## TRIAL METADATA
        trial_duration = target_data['timestamp'].iloc[-1] - target_data['timestamp'].iloc[0]
        trial_nb_frames = len(target_data)
        if device_ID is not None:
            if device_ID == '1944301011EA1F1300':
                cam_position = 'right'
            elif device_ID == '19443010910F481300':
                cam_position = 'left'
        if hand is not None:
            if hand == cam_position:
                cam_hand_position = 'same'
            else:
                cam_hand_position = 'opposite'                
                
        
        ## HAND DETECTION
        nb_hand_found = target_data['task_hand_found'].sum()
        ratio_hand_found = nb_hand_found / trial_nb_frames
        
        #find the first True value in the 'task_hand_found' column
        # idx_first_hand_found = target_data['task_hand_found'].idxmax()
        # first_hand_found_timestamp = target_data.loc[idx_first_hand_found, 'timestamp']
        
        # find the index at wich the hand was detected consecutively for min_number_consecutive_hand_detections frames
        min_number_consecutive_hand_detections = 3
        nb_consecutive_hand_detections = 0
        idx_first_consecutive_hand_found = trial_nb_frames-1
        for index, row in target_data.iterrows():
            if row['task_hand_found'] == True:
                nb_consecutive_hand_detections += 1
            else:
                nb_consecutive_hand_detections = 0
            if nb_consecutive_hand_detections == min_number_consecutive_hand_detections:
                idx_first_consecutive_hand_found = index-min_number_consecutive_hand_detections+1
                break        
        
        first_consecutive_hand_found_timestamp = target_data.loc[idx_first_consecutive_hand_found, 'timestamp']
        
        #check if the hand was continuously estimated
        is_hand_estimation_continuous = target_data.loc[idx_first_consecutive_hand_found:,'task_hand_found'].all()
        
        ## OBJECT DETECTION
        
        nb_object_found = target_data['task_object_found'].sum()
        ratio_object_found = target_data['task_object_found'].sum() / trial_nb_frames
        
        #find the first True value in the 'task_object_found' column
        # idx_first_object_found = target_data['task_object_found'].idxmax()
        # first_object_found_timestamp = target_data.loc[idx_first_object_found, 'timestamp']
        
        # find the index at wich the object was detected consecutively for min_number_consecutive_object_detections frames
        idx_first_consecutive_object_found = trial_nb_frames-1
        min_number_consecutive_object_detections = 3
        nb_consecutive_object_detections = 0
        for index, row in target_data.iterrows():
            if row['task_object_found'] == True:
                nb_consecutive_object_detections += 1
            else:
                nb_consecutive_object_detections = 0
            if nb_consecutive_object_detections == min_number_consecutive_object_detections:
                idx_first_consecutive_object_found = index-min_number_consecutive_object_detections+1
                break
        first_consecutive_object_found_timestamp = target_data.loc[idx_first_consecutive_object_found, 'timestamp']
        
        #check the column 'task_object_estimation_continuous' to see if the object was continuously estimated
        is_object_estimation_continuous = target_data.loc[idx_first_consecutive_object_found:,'task_object_found'].all()
        
        ## TRIAL VALIDATION
        
        # achievable_index = max(idx_first_hand_found, idx_first_object_found)
        achievable_index = max(idx_first_consecutive_hand_found, idx_first_consecutive_object_found)
        if achievable_index == trial_nb_frames-1:
            return None
        achievable_timestamp = target_data.loc[achievable_index, 'timestamp']
        achievable_time = trial_duration - achievable_timestamp 
        nb_achievable_task_detections = trial_nb_frames - achievable_index
        ratio_achievable_task_detections = nb_achievable_task_detections / trial_nb_frames

        # check hand detection after achievable_index
        max_nb_consecutive_failed_hand_detections = 3
        too_many_consecutive_failed_hand_detections = False
        nb_consecutive_failed_hand_detections =0
        for index in range(achievable_index, trial_nb_frames):
            row = target_data.iloc[index]
            if row['task_hand_found'] == True :
                nb_consecutive_failed_hand_detections = 0
            else:
                nb_consecutive_failed_hand_detections += 1
            if nb_consecutive_failed_hand_detections > max_nb_consecutive_failed_hand_detections:
                too_many_consecutive_failed_hand_detections = True
                break
        
        # check object detection after achievable_index
        max_nb_consecutive_failed_object_detections = 3
        too_many_consecutive_failed_object_detections = False
        nb_consecutive_failed_object_detections = 0
        for index in range(achievable_index, trial_nb_frames):
            row = target_data.iloc[index]
            if row['task_object_found'] == True :
                nb_consecutive_failed_object_detections = 0
            else:
                nb_consecutive_failed_object_detections += 1
            if nb_consecutive_failed_object_detections > max_nb_consecutive_failed_object_detections:
                too_many_consecutive_failed_object_detections = True
                break
        
        # define the minimum ratio of hand and object found to consider the trial valid
        min_ratio_hand_found = 0.5
        min_ratio_object_found = 0.5
        min_ratio_achievable_task_detections = 0.4
        min_achievable_time = 0.5
        
        hand_found_ok = ratio_hand_found > min_ratio_hand_found
        object_found_ok = ratio_object_found > min_ratio_object_found
        achievable_task_detections_ok = ratio_achievable_task_detections > min_ratio_achievable_task_detections
        achievable_time_absolute_ok = achievable_time > min_achievable_time
        achievable_time_relative_ok = achievable_time > 0.5 * trial_duration
        
        # is_trial_valid =  hand_found_ok and object_found_ok and achievable_task_detections_ok and achievable_time_ok and not too_many_consecutive_failed_hand_detections and not too_many_consecutive_failed_object_detections
        is_trial_valid = (achievable_time_absolute_ok or achievable_time_relative_ok) and not too_many_consecutive_failed_hand_detections and not too_many_consecutive_failed_object_detections
        # is_trial_valid = ratio_hand_found > 0.7 and ratio_object_found > 0.9 and not too_many_consecutive_failed_hand_detections and not too_many_consecutive_failed_object_detections

        not_valid_reasons = []
        valid_reasons = []
        
        # if not hand_found_ok:
        #     not_valid_reasons.append('hand not found enough')
        # else:
        #     valid_reasons.append('hand found enough')
            
        # if not object_found_ok:
        #     not_valid_reasons.append('object not found enough')
        # else:
        #     valid_reasons.append('object found enough')
        
        if not achievable_time_absolute_ok:
            not_valid_reasons.append('achievable absolute time not enough')
        else:
            valid_reasons.append('achievable absolute time enough')
        
        if not achievable_time_relative_ok:
            not_valid_reasons.append('achievable relative time not enough')
        else:
            valid_reasons.append('achievable relative time enough')
        
        # if not achievable_task_detections_ok:
        #     not_valid_reasons.append('not enough achievable task detections')
        # else:
        #     valid_reasons.append('enough achievable task detections')
        
        if too_many_consecutive_failed_hand_detections:
            not_valid_reasons.append('too many consecutive failed hand detections')
        else:
            valid_reasons.append('not too many consecutive failed hand detections')
            
        if too_many_consecutive_failed_object_detections:
            not_valid_reasons.append('too many consecutive failed object detections')
        else:
            valid_reasons.append('not too many consecutive failed object detections')
        
        
        # create a new column 'task_estimation_achievable' with False before achievable_index and True after
        target_data['task_estimation_achievable'] = False
        target_data.loc[achievable_index:, 'task_estimation_achievable'] = True
              
        ## TARGET IDENTIFICATION
        target_found_label = 'task_target_found'
        target_data[f'target_from_max_metric_correct'] = target_data['target_max_metric_confidence'].eq(target)
        # target_found_label = 'target_from_max_metric_correct'
        
        #find the starting index of the longest series of consecutive True values in the 'task_target_found' column
        idx_start_longest_consecutive_target_found = 0
        longest_series_length = 0
        idx_start_series = 0
        series_length = 0
        previous_target_found = False
        for index, row in target_data.iterrows():
            current_target_found = row[target_found_label]
            if current_target_found == True:
                
                if previous_target_found == False:
                    idx_start_series = index
                series_length += 1
                
                if series_length > longest_series_length:
                    longest_series_length = series_length
                    idx_start_longest_consecutive_target_found = idx_start_series
                previous_target_found = True
            else:
                series_length = 0
                previous_target_found = False
                
            
        
        #find the first True value in the 'task_object_found' column
        idx_first_target_found = target_data[target_found_label].idxmax() 
        idx_first_target_found = idx_start_longest_consecutive_target_found
        
        first_target_found_timestamp = target_data.loc[idx_first_target_found, 'timestamp']
        
        #count the number of True values in the 'task_object_found' column
        nb_target_found = target_data[target_found_label].sum()
        
        # compute the ratio of target found
        target_found_ratio = nb_target_found / nb_achievable_task_detections
        
        # count the number of False values in the 'task_object_found' column after the first True value
        target_data_after_first_target_found = target_data.loc[idx_first_target_found:, target_found_label]
        ratio_target_switch = (len(target_data_after_first_target_found) - target_data_after_first_target_found.sum())/len(target_data_after_first_target_found)
        task_target_idenfication_successful = ratio_target_switch < 0.3
        
        
        
        
        # for each metric, check if colmns target_from_impacts	target_from_distance, target_from_distance_derivative and target_from_direction are have same value as target
        
        metrics = ['impacts', 'distance', 'distance_derivative', 'direction']
        metrics = ['impacts', 'distance', 'future_distance', 'distance_derivative', 'direction']
        for metric in metrics:
            target_data[f'target_from_{metric}_correct'] = target_data[f'target_from_{metric}'].eq(target)
        
        nb_metric_correct = {}
        nb_correct_when_most_probable_wrong = {}
        nb_correct_when_most_probable_wrong_beginning = {}
        nb_correct_when_most_probable_wrong_end = {}
        for metric in metrics:
            nb_metric_correct[metric] = target_data[f'target_from_{metric}_correct'].sum()
            target_data[f'target_from_{metric}_correct_when_most_probable_wrong'] = False
        most_trustworthy_metric = max(nb_metric_correct, key=nb_metric_correct.get)
        nb_metric_correct['max_metric'] = target_data[f'target_from_max_metric_correct'].sum()
        
        # create a new column 'task_target_not_found_but_individual_metrics_correct' with True if the target was not found but at least one of the individual metrics is correct
        target_data['task_target_not_found_but_individual_metrics_correct'] = False
        target_data['task_target_not_found_but_metrics_correct'] = ''
        for index, row in target_data.iterrows():
            if row[target_found_label] == False:
                correct_metrics = []
                for metric in metrics:
                    if row[f'target_from_{metric}_correct']:
                        correct_metrics.append(metric)
                        target_data.loc[index, 'task_target_not_found_but_individual_metrics_correct'] = True
                        target_data.loc[index, f'target_from_{metric}_correct_when_most_probable_wrong'] = True
                target_data.loc[index, 'task_target_not_found_but_metrics_correct'] = ', '.join(correct_metrics)
                
        for metric in metrics:
            nb_correct_when_most_probable_wrong[metric] = target_data[f'target_from_{metric}_correct_when_most_probable_wrong'].sum()
            nb_correct_when_most_probable_wrong_beginning[metric] = target_data.loc[target_data['movement_part'].eq('begin'), f'target_from_{metric}_correct_when_most_probable_wrong'].sum()
            nb_correct_when_most_probable_wrong_end[metric] = target_data.loc[target_data['movement_part'].eq('end'), f'target_from_{metric}_correct_when_most_probable_wrong'].sum()
                        
        # count the number of True values in the 'task_target_not_found_but_individual_metrics_correct' column
        nb_target_not_found_but_individual_metrics_correct = target_data['task_target_not_found_but_individual_metrics_correct'].sum()
        
        # compute the ratio of target not found but individual metrics correct
        ratio_target_not_found_but_individual_metrics_correct = nb_target_not_found_but_individual_metrics_correct / nb_achievable_task_detections
        
        
        #find the starting index of the longest series of consecutive True values in the 'task_target_found' column
        task_target_idenfication_successful_met={}
        first_target_found_timestamp_met = {}
        idx_first_target_found_met = {}
        
        metrics = ['impacts', 'distance', 'future_distance', 'distance_derivative', 'max_metric']
        for metric in metrics:
            target_found_label = f'target_from_{metric}_correct'
            idx_start_longest_consecutive_target_found_met = 0
            longest_series_length = 0
            idx_start_series = 0
            series_length = 0
            previous_target_found = False
            for index, row in target_data.iterrows():
                current_target_found = row[target_found_label]
                if current_target_found == True:
                    
                    if previous_target_found == False:
                        idx_start_series = index
                    series_length += 1
                    
                    if series_length > longest_series_length:
                        longest_series_length = series_length
                        idx_start_longest_consecutive_target_found_met = idx_start_series
                    previous_target_found = True
                else:
                    series_length = 0
                    previous_target_found = False
                    
                
            
            #find the first True value in the 'task_object_found' column
            # idx_first_target_found_met = target_data[target_found_label].idxmax() 
            idx_first_target_found_met[metric] = idx_start_longest_consecutive_target_found_met
            
            first_target_found_timestamp_met[metric] = target_data.loc[idx_first_target_found_met[metric], 'timestamp']
            
            #count the number of True values in the 'task_object_found' column
            nb_target_found_met = target_data[target_found_label].sum()
            
            # compute the ratio of target found
            target_found_ratio_met = nb_target_found_met / nb_achievable_task_detections
            
            # count the number of False values in the 'task_object_found' column after the first True value
            target_data_after_first_target_found_met = target_data.loc[idx_first_target_found_met[metric]:, target_found_label]
            ratio_target_switch_met = (len(target_data_after_first_target_found_met) - target_data_after_first_target_found_met.sum())/len(target_data_after_first_target_found_met)
            task_target_idenfication_successful_met[metric] = ratio_target_switch_met < 0.3
        
        ## GRIP IDENTIFICATION
        
        #find the starting index of the longest series of consecutive True values in the 'task_grip_found' column
        idx_start_longest_consecutive_grip_found = 0
        longest_series_length = 0
        idx_start_series = 0
        series_length = 0
        previous_grip_found = False
        for index, row in target_data.iterrows():
            current_grip_found = row['task_grip_found']
            if current_grip_found == True:
                
                if previous_grip_found == False:
                    idx_start_series = index
                series_length += 1
                
                if series_length > longest_series_length:
                    longest_series_length = series_length
                    idx_start_longest_consecutive_grip_found = idx_start_series
                previous_grip_found = True
            else:
                series_length = 0
                previous_grip_found = False
        
        #find the first True value in the 'task_grip_found' column
        first_grip_found_idx = target_data['task_grip_found'].idxmax() 
        first_grip_found_idx = idx_start_longest_consecutive_grip_found
        first_grip_found_timestamp = target_data.loc[first_grip_found_idx, 'timestamp']
        
        #count the number of True values in the 'task_grip_found' column
        nb_grips_found = target_data['task_grip_found'].sum()
        
        #compute the ratio of grip found
        grip_found_ratio = nb_grips_found / nb_achievable_task_detections
        
        # count the number of False values in the 'task_grip_found' column after the first True value
        after_first_grip_found = target_data.loc[first_grip_found_idx:, 'task_grip_found']   
        ratio_grip_switch = (len(after_first_grip_found) - after_first_grip_found.sum())/len(after_first_grip_found)
        task_grip_idenfication_successful = ratio_grip_switch < 0.3
        
        
        # first_grip_found_timestamp_met = {}
        # nb_grips_found_met = {}
        # grip_found_ratio_met = {}
        # task_grip_idenfication_successful_met = {}
        # for metric in metrics:
        #     grip_label = f'task_grip_found_{metric}'
        #     idx_start_longest_consecutive_grip_found = 0
        #     longest_series_length = 0
        #     idx_start_series = 0
        #     series_length = 0
        #     previous_grip_found = False
        #     for index, row in target_data.iterrows():
        #         current_grip_found = row[grip_label]
        #         if current_grip_found == True:
                    
        #             if previous_grip_found == False:
        #                 idx_start_series = index
        #             series_length += 1
                    
        #             if series_length > longest_series_length:
        #                 longest_series_length = series_length
        #                 idx_start_longest_consecutive_grip_found = idx_start_series
        #             previous_grip_found = True
        #         else:
        #             series_length = 0
        #             previous_grip_found = False
            
        #     #find the first True value in the 'task_grip_found' column
        #     first_grip_found_idx = target_data[grip_label].idxmax() 
        #     first_grip_found_idx = idx_start_longest_consecutive_grip_found
        #     first_grip_found_timestamp_met[metric] = target_data.loc[first_grip_found_idx, 'timestamp']
            
        #     #count the number of True values in the 'task_grip_found' column
        #     nb_grips_found_met[metric] = target_data[grip_label].sum()
            
        #     #compute the ratio of grip found
        #     grip_found_ratio_met[metric] = nb_grips_found_met[metric] / nb_achievable_task_detections
            
        #     # count the number of False values in the 'task_grip_found' column after the first True value
        #     after_first_grip_found = target_data.loc[first_grip_found_idx:, grip_label]   
        #     ratio_grip_switch_met = (len(after_first_grip_found) - after_first_grip_found.sum())/len(after_first_grip_found)
        #     task_grip_idenfication_successful_met[metric] = ratio_grip_switch_met < 0.3
        
        
        ## KINEMATIC DATA
        # print(f"target_data: {target_data}")
        # print(f"target_data.columns: {target_data.columns}")
        # print(f'target_data["hand_scalar_velocity"]: {target_data["hand_scalar_velocity"]}')
        # get maximum velocity value and index in 'hand_scalar_velocity' column
        idx_max_velocity = target_data['hand_scalar_velocity'].idxmax()
        max_velocity = target_data.loc[idx_max_velocity, 'hand_scalar_velocity']
        
        target_data_after_max_vel = target_data[idx_max_velocity:]
        target_data_before_max_vel = target_data[:idx_max_velocity]
        
        # get the minimum value and index in 'hand_scalar_velocity' column after the maximum velocity
        idx_min_velocity = target_data_after_max_vel['hand_scalar_velocity'].idxmin()
        min_velocity = target_data_after_max_vel.loc[idx_min_velocity, 'hand_scalar_velocity']
        
        delta_velocity = max_velocity - min_velocity
        min_vel_ratio = 0.1
        velocity_threshold_end = min_velocity + min_vel_ratio * delta_velocity
        velocity_threshold_beginning = min_vel_ratio * max_velocity
        
        #find the first index where the velocity is above the threshold
        idx_begin_movement = target_data_before_max_vel['hand_scalar_velocity'].gt(velocity_threshold_beginning).idxmax()
        begin_movement_timestamp = target_data.loc[idx_begin_movement, 'timestamp']
        
        
        # find the first index where the velocity is below the threshold
        idx_end_movement = target_data_after_max_vel['hand_scalar_velocity'].lt(velocity_threshold_end).idxmax()
        end_movement_timestamp = target_data.loc[idx_end_movement, 'timestamp']
        resting_time = trial_duration - end_movement_timestamp
        delay_time = min(begin_movement_timestamp, achievable_timestamp)
        
        movement_duration = end_movement_timestamp - begin_movement_timestamp
        target_found_margin = end_movement_timestamp - first_target_found_timestamp
        grip_found_margin = end_movement_timestamp - first_grip_found_timestamp
        target_found_delay = first_target_found_timestamp - delay_time
        grip_found_delay = first_grip_found_timestamp - delay_time
        
        target_found_margin_met = {}
        grip_found_margin_met = {}
        target_found_delay_met = {}
        grip_found_delay_met = {}
        
        for metric in metrics:
            target_found_margin_met[metric] = end_movement_timestamp - first_target_found_timestamp_met[metric]
            target_found_delay_met[metric] = first_target_found_timestamp_met[metric] - delay_time
            # grip_found_margin_met[metric] = end_movement_timestamp - first_grip_found_timestamp_met[metric]
            # grip_found_delay_met[metric] = first_grip_found_timestamp_met[metric] - delay_time
            
        
        
        
        ## TIME TO TARGET ESTIMATION
        
        # get sub-df with only the rows from the first target found
        target_data_after_first_target_found = target_data.loc[idx_first_target_found:]
        idx_end_movement_after_target = idx_end_movement - idx_first_target_found
        
        na_values = [rt.NO_SIGN_SWITCH, rt.NO_POLY_FIT, rt.NO_REAL_POSITIVE_ROOT]
        for index, row in target_data_after_first_target_found.iterrows():
            if row['estimated_target_time_to_impact'] in na_values:
                target_data_after_first_target_found.loc[index, 'estimated_target_time_to_impact'] = np.nan
        
        print(f"idx_end_movement_after_target: {idx_end_movement_after_target}")
        print(f"len(target_data_after_first_target_found): {len(target_data_after_first_target_found)}")
        # add a new column 'time_to_target' with the time to target, compute the difference between the timestamp and the timestamp of the last row
        if idx_end_movement_after_target < 0:
            target_data_after_first_target_found['time_to_target'] = np.nan
        else:
            target_data_after_first_target_found['time_to_target'] = target_data_after_first_target_found['timestamp'] - target_data_after_first_target_found['timestamp'].iloc[idx_end_movement_after_target]
        
        #check
        
        # add a new column 'time_to_target_error' with the difference between the time to target estimation and the actual time to target
        target_data_after_first_target_found['time_to_target_error'] = target_data_after_first_target_found['time_to_target'] - target_data_after_first_target_found['estimated_target_time_to_impact']
        
        # compute the rmse of the time to target estimation
        time_to_target_rmse = np.sqrt(np.mean(target_data_after_first_target_found['time_to_target_error']**2))
        
        evaluation_data = {'task_target': target,
                           'task_grip': grip,   
                            'cam_hand_position': cam_hand_position,
                           'movement_mode': mode,
                           'trial_nb_frames': trial_nb_frames,
                           'trial_nb_achievable_task_detections': nb_achievable_task_detections,
                           
                           'idx_first_hand_found': idx_first_consecutive_hand_found,
                           'nb_hand_found': nb_hand_found,
                           'ratio_hand_found': ratio_hand_found,
                           'is_hand_estimation_continuous': is_hand_estimation_continuous,
                           
                           'idx_first_object_found': idx_first_consecutive_object_found,
                           'nb_object_found': nb_object_found,
                           'ratio_object_found': ratio_object_found,
                           'is_object_estimation_continuous': is_object_estimation_continuous,
                           
                           'first_hand_found': first_consecutive_hand_found_timestamp,
                           'first_object_found': first_consecutive_object_found_timestamp,                           
                           'first_target_found': first_target_found_timestamp, 
                           
                           
                           'is_trial_valid' : is_trial_valid, 
                           'not_valid_reasons': not_valid_reasons,
                           'valid_reasons': valid_reasons,
                           
                           'nb_target_found': nb_target_found, 
                            'nb_impacts_correct': nb_metric_correct['impacts'],
                            'nb_distance_correct': nb_metric_correct['distance'],
                            'nb_future_distance_correct': nb_metric_correct['future_distance'],
                            'nb_distance_derivative_correct': nb_metric_correct['distance_derivative'],
                            'nb_direction_correct': nb_metric_correct['direction'],
                            'nb_max_metric_correct': nb_metric_correct['max_metric'],
                            
                            'nb_impacts_correct_when_most_probable_wrong': nb_correct_when_most_probable_wrong['impacts'],
                            'nb_distance_correct_when_most_probable_wrong': nb_correct_when_most_probable_wrong['distance'],
                            'nb_future_distance_correct_when_most_probable_wrong': nb_correct_when_most_probable_wrong['future_distance'],
                            'nb_distance_derivative_correct_when_most_probable_wrong': nb_correct_when_most_probable_wrong['distance_derivative'],
                            'nb_direction_correct_when_most_probable_wrong': nb_correct_when_most_probable_wrong['direction'],
                            
                            'nb_impacts_correct_when_most_probable_wrong_beginning': nb_correct_when_most_probable_wrong_beginning['impacts'],
                            'nb_distance_correct_when_most_probable_wrong_beginning': nb_correct_when_most_probable_wrong_beginning['distance'],
                            'nb_future_distance_correct_when_most_probable_wrong_beginning': nb_correct_when_most_probable_wrong_beginning['future_distance'],
                            'nb_distance_derivative_correct_when_most_probable_wrong_beginning': nb_correct_when_most_probable_wrong_beginning['distance_derivative'],
                            'nb_direction_correct_when_most_probable_wrong_beginning': nb_correct_when_most_probable_wrong_beginning['direction'],
                            
                            'nb_impacts_correct_when_most_probable_wrong_end': nb_correct_when_most_probable_wrong_end['impacts'],
                            'nb_distance_correct_when_most_probable_wrong_end': nb_correct_when_most_probable_wrong_end['distance'],
                            'nb_future_distance_correct_when_most_probable_wrong_end': nb_correct_when_most_probable_wrong_end['future_distance'],
                            'nb_distance_derivative_correct_when_most_probable_wrong_end': nb_correct_when_most_probable_wrong_end['distance_derivative'],
                            'nb_direction_correct_when_most_probable_wrong_end': nb_correct_when_most_probable_wrong_end['direction'],
                            
                            'most_trustworthy_metric': most_trustworthy_metric,
                            'ratio_target_not_found_but_individual_metrics_correct': ratio_target_not_found_but_individual_metrics_correct,
                            'nb_target_not_found_but_individual_metrics_correct': nb_target_not_found_but_individual_metrics_correct,
                            
                           'target_found_ratio': target_found_ratio, 
                           'idx_first_target_found': idx_first_target_found,
                           'idx_start_longest_consecutive_target_found': idx_start_longest_consecutive_target_found,
                           'ratio_target_switch': ratio_target_switch, 
                           
                           'task_target_idenfication_successful': task_target_idenfication_successful,
                           'task_target_idenfication_successful_distance': task_target_idenfication_successful_met['distance'],
                            'task_target_idenfication_successful_distance_derivative': task_target_idenfication_successful_met['distance_derivative'],
                            'task_target_idenfication_successful_impacts': task_target_idenfication_successful_met['impacts'],
                            'task_target_idenfication_successful_future_distance': task_target_idenfication_successful_met['future_distance'],
                            'task_target_idenfication_successful_max_metric': task_target_idenfication_successful_met['max_metric'],
                           
                            
                           
                           'nb_grips_found': nb_grips_found,
                           'grip_found_ratio': grip_found_ratio,
                           'ratio_grip_switch': ratio_grip_switch,
                           
                           'task_grip_idenfication_successful': task_grip_idenfication_successful,
                        #    'task_grip_idenfication_successful_distance': task_grip_idenfication_successful_met['distance'],
                        #     'task_grip_idenfication_successful_distance_derivative': task_grip_idenfication_successful_met['distance_derivative'],
                        #     'task_grip_idenfication_successful_impacts': task_grip_idenfication_successful_met['impacts'],
                        #     'task_grip_idenfication_successful_future_distance': task_grip_idenfication_successful_met['future_distance'],
                        #     'task_grip_idenfication_successful_max_metric': task_grip_idenfication_successful_met['max_metric'],
                           
                           'max_velocity': max_velocity,
                           'min_velocity': min_velocity,
                           'delta_velocity': delta_velocity,
                           'velocity_threshold_beginning': velocity_threshold_beginning,
                            'velocity_threshold_end': velocity_threshold_end,
                            
                           'trial_duration': trial_duration,
                           'movement_duration': movement_duration,
                           'begin_movement_timestamp': begin_movement_timestamp,
                           'achievable_timestamp': achievable_timestamp,
                            'end_movement_timestamp': end_movement_timestamp,  
                           'grip_found_margin': grip_found_margin,
                            'grip_found_delay': grip_found_delay,
                            
                            'target_found_timestamp_global': first_target_found_timestamp,
                            'target_found_timestamp_distance': first_target_found_timestamp_met['distance'],
                            'target_found_timestamp_distance_derivative': first_target_found_timestamp_met['distance_derivative'],
                            'target_found_timestamp_impacts': first_target_found_timestamp_met['impacts'],
                            'target_found_timestamp_future_distance': first_target_found_timestamp_met['future_distance'],
                            'target_found_timestamp_max_metric': first_target_found_timestamp_met['max_metric'],
                            'first_grip_found': first_grip_found_timestamp, 
                            
                            'target_found_index_global' : idx_first_target_found,
                            'target_found_index_distance':idx_first_target_found_met['distance'],
                            'target_found_index_distance_derivative':idx_first_target_found_met['distance_derivative'],
                            'target_found_index_impacts':idx_first_target_found_met['impacts'],
                            'target_found_index_future_distance':idx_first_target_found_met['future_distance'],
                            'target_found_index_max_metric':idx_first_target_found_met['max_metric'],
                            
                           'target_found_margin': target_found_margin,         
                           'target_found_delay': target_found_delay,   
                           'target_found_margin_distance': target_found_margin_met['distance'],              
                           'target_found_delay_distance': target_found_delay_met['distance'],
                           'target_found_margin_future_distance': target_found_margin_met['future_distance'],              
                           'target_found_delay_future_distance': target_found_delay_met['future_distance'],
                           'target_found_margin_distance_derivative': target_found_margin_met['distance_derivative'],              
                           'target_found_delay_distance_derivative': target_found_delay_met['distance_derivative'],
                            'target_found_margin_max_metric': target_found_margin_met['max_metric'],
                            'target_found_delay_max_metric': target_found_delay_met['max_metric'],   
                           'target_found_margin_impacts': target_found_margin_met['impacts'],              
                           'target_found_delay_impacts': target_found_delay_met['impacts'],
                           
                        #    'grip_found_margin_distance': grip_found_margin_met['distance'],
                        #     'grip_found_delay_distance': grip_found_delay_met['distance'],                            
                        #    'grip_found_margin_future_distance': grip_found_margin_met['future_distance'],
                        #     'grip_found_delay_future_distance': grip_found_delay_met['future_distance'],                            
                        #    'grip_found_margin_distance_derivative': grip_found_margin_met['distance_derivative'],
                        #     'grip_found_delay_distance_derivative': grip_found_delay_met['distance_derivative'],                        
                        #    'grip_found_margin_impacts': grip_found_margin_met['impacts'],
                        #     'grip_found_delay_impacts': grip_found_delay_met['impacts'],                            
                        #     'grip_found_margin_max_metric': grip_found_margin_met['max_metric'],
                        #     'grip_found_delay_max_metric': grip_found_delay_met['max_metric'],
                            
                            'resting_time': resting_time,
                           'time_to_target_rmse': time_to_target_rmse
                           }
        
        return evaluation_data
    
    def plot_target_data(target_data, hand, save_path, device_ID, target,mode):     # 16_05_save
        if 'most_probable_target' not in target_data.columns:
            return
        fig, axs = plt.subplots(8, 1, figsize=(3., 8.), sharex=True)
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.size"] = 8  
        time = target_data['timestamp']
        x = time
        
        #scalar velocity
        if 'hand_scalar_velocity' in target_data.columns:
            y = target_data['hand_scalar_velocity']
            axs[0].plot(x, y, label='scalar velocity', color='black')
            # axs[0].set_title('scalar velocity')
            # axs[0].set_xlabel('time')
            axs[0].set_ylabel('Hand scalar \nvelocity (mm/s)')
            # axs[0].legend()
        #distances
        
        expected_objects = sc.RigidObject.LABEL_EXPE_NAMES
        legend_handles = []
        for obj in expected_objects.values():
            # transform first letter to uppercase
            lab_obj = obj[0].upper() + obj[1:]
            legend_handles.append(mlines.Line2D([], [], color=RigidObject._TARGETS_COLORS_DICT[obj], label=lab_obj))  
        # metrics = [ 'distance_derivative', 'impacts', 'distance', 'future_distance', 'max_metric','global']
        # metrics_for_labels = ['Distance derivative', 'Impacts', 'Distance', 'Future distance','Maximum', 'Global']
        metrics = [ 'distance_derivative', 'impacts', 'distance', 'future_distance','global']
        metrics_for_labels = ['Distance derivative', 'Impacts', 'Distance', 'Future distance', 'Global']
        for i, metric in enumerate(metrics):
            row = i + 1
            for obj in expected_objects.values():
                col_title = f'{metric}_confidence_{obj}'
                if col_title not in target_data.columns:
                    continue
                x = time
                y = target_data[col_title]
                axs[row].plot(x, y, label=obj, color=RigidObject._TARGETS_COLORS_DICT[obj])
            axs[row].set_ylabel(f'{metrics_for_labels[i]}\nconfidence')
            if row > 1:
                axs[row].set_ylim([0, 1])
                
        ### TGARGET IDENTIFICATION
        target_time = target_data['most_probable_target']
        target_time_value = []
        for i in range(len(time)):
            found = False
            for k, ob in enumerate(expected_objects.keys()):
                if target_time[i] == expected_objects[ob]:
                    target_time_value.append(k+1)
                    found = True
            if not found:
                target_time_value.append(target_time[i])
        
        width = 2
        current_target = target_time_value[0]
        current_time=[]
        current_target_time_value=[]
        if target_time[0] in RigidObject._TARGETS_COLORS_DICT.keys():
            current_color = RigidObject._TARGETS_COLORS_DICT[target_time[0]]
        else:
            current_color = 'black'
        
        for i in range(len(time)):
            print(f'target_time_value[{i}]: {target_time_value[i]}')
            print(f'current_target: {current_target}')
            if target_time_value[i] == current_target:
                current_time.append(time[i])
                current_target_time_value.append(target_time_value[i])
            else:
                print('next target time part')
                current_target = target_time_value[i]
                axs[6].plot(current_time, current_target_time_value, drawstyle='steps-post', color=current_color, linewidth=width)
                if target_time[i] in RigidObject._TARGETS_COLORS_DICT.keys():
                    current_color = RigidObject._TARGETS_COLORS_DICT[target_time[i]]
                else:
                    current_color = 'black'
                print('plot target time part')
                current_time = [time[i]]
                current_target_time_value = [target_time_value[i]]
        axs[6].plot(current_time, current_target_time_value, drawstyle='steps-post', color=current_color, linewidth=width)
        print('plot last target time part')
            
            
        # axs[6].plot(time, target_time_value , drawstyle='steps-post', color='black')
        axs[6].set_yticks([1,2,3,4], expected_objects.values())
        axs[6].set_ylim([0, 5])
        axs[6].set_ylabel('Most probable \ntarget')
        
        # 
        Ticks = expected_objects.values()
        # set first letter to uppercase
        Ticks = [obj[0].upper() + obj[1:] for obj in Ticks]
        axs[6].set_yticks([1,2,3,4], Ticks)
        #rotate y labels 45 degrees
        # plt.setp(axs[6].get_yticklabels(), rotation=45, ha='right', va='center_baseline'
        #             , rotation_mode='anchor')
        
        grip_time = target_data['grip']
        grips = ['palmar', 'pinch']
        colors = {'palmar':'blue',
                  'pinch': 'magenta'}
        grip_time_value = []
        for i in range(len(grip_time)):
            found = False
            for k, ob in enumerate(grips):
                if grip_time[i] == grips[k]:
                    grip_time_value.append(k+1)
                    found = True
            if not found:
                grip_time_value.append(grip_time[i])
                
        
        current_grip = grip_time_value[0]
        current_time=[]
        current_grip_time_value=[]
        if grip_time[0] in grips:
            current_color = colors[grips[current_grip-1]]
        else:
            current_color = 'black'
        
        for i in range(len(time)):
            print(f'grip_time_value[{i}]: {grip_time_value[i]}')
            print(f'current_grip: {current_grip}')
            if grip_time_value[i] == current_grip:
                current_time.append(time[i])
                current_grip_time_value.append(grip_time_value[i])
            else:
                print('next grip time part')
                current_grip = grip_time_value[i]
                axs[7].plot(current_time, current_grip_time_value, drawstyle='steps-post', color=current_color, linewidth=width)
                if grip_time[i] in grips:
                    current_color = colors[grips[current_grip-1]]
                else:
                    current_color = 'black'
                print('plot grip time part')
                current_time = [time[i]]
                current_grip_time_value = [grip_time_value[i]]
        axs[7].plot(current_time, current_grip_time_value, drawstyle='steps-post', color=current_color, linewidth=width)
        print('plot last grip time part')
            
                    
        
        # axs[7].plot(time, grip_time_value,  drawstyle='steps-post', color='black', width=2)
        
        
        tgrips = ['Palmar', 'Pinch']
        axs[7].set_yticks([1,2], tgrips)
        axs[7].set_ylim([0, 3])
        axs[7].set_ylabel('Selected\ngrip')
        
        eval_data = Trial.evaluate_target_data(target_data, device_ID=device_ID, hand=hand, target=target, mode=mode)    
        if eval_data is not None:
            #get target_found_timestamps
            tg_fd_timsps = {}
            tg_fd_idx = {}
            y_annot = {}
            
            for i, metric in enumerate(metrics):
                print(f'tg_fd_timsps: {tg_fd_timsps}')
                print(f'metric: {metric}')
                tg_fd_timsps[metric] = eval_data[f'target_found_timestamp_{metric}']
                tg_fd_idx[metric] = eval_data[f'target_found_index_{metric}']
                tar_col_title = f'{metric}_confidence_{target}'
                ys = target_data[tar_col_title]
                y_annot[metric] = ys[tg_fd_idx[metric]]
            
            for i, metric in enumerate(metrics):
                y_lim = axs[i+1].get_ylim()
                delta_y = abs(y_lim[1] - y_lim[0])
                if y_annot[metric] > y_lim[1]-delta_y/2:
                    y_txt = y_annot[metric] - delta_y/3
                else:
                    y_txt = y_annot[metric] + delta_y/3
                axs[i+1].annotate('Target found', xy=(tg_fd_timsps[metric], y_annot[metric]), xytext=(tg_fd_timsps[metric], y_txt),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=5, headlength=5))
            print(f'timestamps for trial {save_path}: {tg_fd_timsps}')
                
        fig.legend(handles=legend_handles, loc='upper center', ncol=4)
        axs[-1].set_xlabel('Time (s)')
        axs[-1].set_xlim([time.iloc[0], time.iloc[-1]])
        # plt.tight_layout()
        fig.tight_layout()
        plt.subplots_adjust(hspace=0.2, wspace=0.1, top=0.95)
        fig.align_ylabels()
        if save_path is not None:
            plt.savefig(save_path, format='eps', bbox_inches='tight')
            plt.savefig(save_path.replace('.eps', '.png'), format='png', bbox_inches='tight')   
            plt.savefig(save_path.replace('.eps', '.pdf'), format='pdf', bbox_inches='tight')   
        # plt.show()
            
        
    
    def evaluate_target_data_new(target_data, device_ID = None, hand = None ):     
        
        print(f"evaluate target_data: {target_data}")
        ## TRIAL METADATA
        trial_duration = target_data['timestamp'].iloc[-1] - target_data['timestamp'].iloc[0]
        trial_nb_frames = len(target_data)
        if device_ID is not None:
            if device_ID == '1944301011EA1F1300':
                cam_position = 'right'
            elif device_ID == '19443010910F481300':
                cam_position = 'left'
        if hand is not None:
            if hand == cam_position:
                cam_hand_position = 'same'
            else:
                cam_hand_position = 'opposite'                
                
        
        ## HAND DETECTION
        nb_hand_found = target_data['task_hand_found'].sum()
        ratio_hand_found = nb_hand_found / trial_nb_frames
        
        #find the first True value in the 'task_hand_found' column
        # idx_first_hand_found = target_data['task_hand_found'].idxmax()
        # first_hand_found_timestamp = target_data.loc[idx_first_hand_found, 'timestamp']
        
        # find the index at wich the hand was detected consecutively for min_number_consecutive_hand_detections frames
        min_number_consecutive_hand_detections = 3
        nb_consecutive_hand_detections = 0
        idx_first_consecutive_hand_found = trial_nb_frames-1
        for index, row in target_data.iterrows():
            if row['task_hand_found'] == True:
                nb_consecutive_hand_detections += 1
            else:
                nb_consecutive_hand_detections = 0
            if nb_consecutive_hand_detections == min_number_consecutive_hand_detections:
                idx_first_consecutive_hand_found = index-min_number_consecutive_hand_detections+1
                break        
        
        first_consecutive_hand_found_timestamp = target_data.loc[idx_first_consecutive_hand_found, 'timestamp']
        
        #check if the hand was continuously estimated
        is_hand_estimation_continuous = target_data.loc[idx_first_consecutive_hand_found:,'task_hand_found'].all()
        
        ## OBJECT DETECTION
        
        nb_object_found = target_data['task_object_found'].sum()
        ratio_object_found = target_data['task_object_found'].sum() / trial_nb_frames
        
        #find the first True value in the 'task_object_found' column
        # idx_first_object_found = target_data['task_object_found'].idxmax()
        # first_object_found_timestamp = target_data.loc[idx_first_object_found, 'timestamp']
        
        # find the index at wich the object was detected consecutively for min_number_consecutive_object_detections frames
        idx_first_consecutive_object_found = trial_nb_frames-1
        min_number_consecutive_object_detections = 3
        nb_consecutive_object_detections = 0
        for index, row in target_data.iterrows():
            if row['task_object_found'] == True:
                nb_consecutive_object_detections += 1
            else:
                nb_consecutive_object_detections = 0
            if nb_consecutive_object_detections == min_number_consecutive_object_detections:
                idx_first_consecutive_object_found = index-min_number_consecutive_object_detections+1
                break
        first_consecutive_object_found_timestamp = target_data.loc[idx_first_consecutive_object_found, 'timestamp']
        
        #check the column 'task_object_estimation_continuous' to see if the object was continuously estimated
        is_object_estimation_continuous = target_data.loc[idx_first_consecutive_object_found:,'task_object_found'].all()
        
        ## TRIAL VALIDATION
        
        # achievable_index = max(idx_first_hand_found, idx_first_object_found)
        achievable_index = max(idx_first_consecutive_hand_found, idx_first_consecutive_object_found)
        if achievable_index == trial_nb_frames-1:
            return None
        achievable_timestamp = target_data.loc[achievable_index, 'timestamp']
        achievable_time = trial_duration - achievable_timestamp 
        nb_achievable_task_detections = trial_nb_frames - achievable_index
        ratio_achievable_task_detections = nb_achievable_task_detections / trial_nb_frames

        # check hand detection after achievable_index
        max_nb_consecutive_failed_hand_detections = 3
        too_many_consecutive_failed_hand_detections = False
        nb_consecutive_failed_hand_detections =0
        for index in range(achievable_index, trial_nb_frames):
            row = target_data.iloc[index]
            if row['task_hand_found'] == True :
                nb_consecutive_failed_hand_detections = 0
            else:
                nb_consecutive_failed_hand_detections += 1
            if nb_consecutive_failed_hand_detections > max_nb_consecutive_failed_hand_detections:
                too_many_consecutive_failed_hand_detections = True
                break
        
        # check object detection after achievable_index
        max_nb_consecutive_failed_object_detections = 3
        too_many_consecutive_failed_object_detections = False
        nb_consecutive_failed_object_detections = 0
        for index in range(achievable_index, trial_nb_frames):
            row = target_data.iloc[index]
            if row['task_object_found'] == True :
                nb_consecutive_failed_object_detections = 0
            else:
                nb_consecutive_failed_object_detections += 1
            if nb_consecutive_failed_object_detections > max_nb_consecutive_failed_object_detections:
                too_many_consecutive_failed_object_detections = True
                break
        
        # define the minimum ratio of hand and object found to consider the trial valid
        min_ratio_hand_found = 0.5
        min_ratio_object_found = 0.5
        min_ratio_achievable_task_detections = 0.4
        min_achievable_time = 0.5
        
        hand_found_ok = ratio_hand_found > min_ratio_hand_found
        object_found_ok = ratio_object_found > min_ratio_object_found
        achievable_task_detections_ok = ratio_achievable_task_detections > min_ratio_achievable_task_detections
        achievable_time_absolute_ok = achievable_time > min_achievable_time
        achievable_time_relative_ok = achievable_time > 0.5 * trial_duration
        
        # is_trial_valid =  hand_found_ok and object_found_ok and achievable_task_detections_ok and achievable_time_ok and not too_many_consecutive_failed_hand_detections and not too_many_consecutive_failed_object_detections
        # is_trial_valid = (achievable_time_absolute_ok or achievable_time_relative_ok) and not too_many_consecutive_failed_hand_detections and not too_many_consecutive_failed_object_detections
        is_trial_valid = ratio_hand_found > 0.7 and ratio_object_found > 0.9 and not too_many_consecutive_failed_hand_detections and not too_many_consecutive_failed_object_detections

        not_valid_reasons = []
        valid_reasons = []
        
        # if not hand_found_ok:
        #     not_valid_reasons.append('hand not found enough')
        # else:
        #     valid_reasons.append('hand found enough')
            
        # if not object_found_ok:
        #     not_valid_reasons.append('object not found enough')
        # else:
        #     valid_reasons.append('object found enough')
        
        if not achievable_time_absolute_ok:
            not_valid_reasons.append('achievable absolute time not enough')
        else:
            valid_reasons.append('achievable absolute time enough')
        
        if not achievable_time_relative_ok:
            not_valid_reasons.append('achievable relative time not enough')
        else:
            valid_reasons.append('achievable relative time enough')
        
        # if not achievable_task_detections_ok:
        #     not_valid_reasons.append('not enough achievable task detections')
        # else:
        #     valid_reasons.append('enough achievable task detections')
        
        if too_many_consecutive_failed_hand_detections:
            not_valid_reasons.append('too many consecutive failed hand detections')
        else:
            valid_reasons.append('not too many consecutive failed hand detections')
            
        if too_many_consecutive_failed_object_detections:
            not_valid_reasons.append('too many consecutive failed object detections')
        else:
            valid_reasons.append('not too many consecutive failed object detections')
        
        
        # create a new column 'task_estimation_achievable' with False before achievable_index and True after
        target_data['task_estimation_achievable'] = False
        target_data.loc[achievable_index:, 'task_estimation_achievable'] = True
              
        ## TARGET IDENTIFICATION
        
        #find the first True value in the 'task_object_found' column
        idx_first_target_found = target_data['task_target_found'].idxmax() 
        first_target_found_timestamp = target_data.loc[idx_first_target_found, 'timestamp']
        target_found_delay = first_target_found_timestamp - first_consecutive_hand_found_timestamp
        
        #count the number of True values in the 'task_object_found' column
        nb_target_found = target_data['task_target_found'].sum()
        
        # compute the ratio of target found
        target_found_ratio = nb_target_found / nb_achievable_task_detections
        
        # count the number of False values in the 'task_object_found' column after the first True value
        target_data_after_first_target_found = target_data.loc[idx_first_target_found:, 'task_target_found']
        ratio_target_switch = (len(target_data_after_first_target_found) - target_data_after_first_target_found.sum())/len(target_data_after_first_target_found)
        task_target_idenfication_successful = ratio_target_switch < 0.3
        

        
        ## GRIP IDENTIFICATION
        
        #find the first True value in the 'task_grip_found' column
        first_grip_found = target_data['task_grip_found'].idxmax() 
        first_grip_found_timestamp = target_data.loc[first_grip_found, 'timestamp']
        grip_found_delay = first_grip_found_timestamp - first_consecutive_hand_found_timestamp
        
        #count the number of True values in the 'task_grip_found' column
        nb_grips_found = target_data['task_grip_found'].sum()
        
        #compute the ratio of grip found
        grip_found_ratio = nb_grips_found / nb_achievable_task_detections
        
        # count the number of False values in the 'task_grip_found' column after the first True value
        after_first_grip_found = target_data.loc[first_grip_found:, 'task_grip_found']   
        ratio_grip_switch = (len(after_first_grip_found) - after_first_grip_found.sum())/len(after_first_grip_found)
        task_grip_idenfication_successful = ratio_grip_switch < 0.3
        
        ## KINEMATIC DATA
        print(f"target_data: {target_data}")
        print(f"target_data.columns: {target_data.columns}")
        print(f'target_data["hand_scalar_velocity"]: {target_data["hand_scalar_velocity"]}')
        # get maximum velocity value and index in 'hand_scalar_velocity' column
        idx_max_velocity = target_data['hand_scalar_velocity'].idxmax()
        max_velocity = target_data.loc[idx_max_velocity, 'hand_scalar_velocity']
        
        target_data_after_max_vel = target_data[idx_max_velocity:]
        
        # get the minimum value and index in 'hand_scalar_velocity' column after the maximum velocity
        idx_min_velocity = target_data_after_max_vel['hand_scalar_velocity'].idxmin()
        min_velocity = target_data_after_max_vel.loc[idx_min_velocity, 'hand_scalar_velocity']
        
        delta_velocity = max_velocity - min_velocity
        min_vel_ratio = 0.1
        min_velocity_threshold = min_velocity + min_vel_ratio * delta_velocity
        
        # find the first index where the velocity is below the threshold
        idx_end_movement = target_data_after_max_vel['hand_scalar_velocity'].lt(min_velocity_threshold).idxmax()
        end_movement_timestamp = target_data.loc[idx_end_movement, 'timestamp']
        resting_time = trial_duration - end_movement_timestamp
        
        
        
        
        ## TIME TO TARGET ESTIMATION
        
        # get sub-df with only the rows from the first target found
        target_data_after_first_target_found = target_data.loc[idx_first_target_found:]
        idx_end_movement_after_target = idx_end_movement - idx_first_target_found
        print(f"idx_end_movement_after_target: {idx_end_movement_after_target}")
        print(f"len(target_data_after_first_target_found): {len(target_data_after_first_target_found)}")
        
        na_values = [rt.NO_SIGN_SWITCH, rt.NO_POLY_FIT, rt.NO_REAL_POSITIVE_ROOT]
        for index, row in target_data_after_first_target_found.iterrows():
            if row['estimated_target_time_to_impact'] in na_values:
                target_data_after_first_target_found.loc[index, 'estimated_target_time_to_impact'] = np.nan
        
        # add a new column 'time_to_target' with the time to target, compute the difference between the timestamp and the timestamp of the last row
        target_data_after_first_target_found['time_to_target'] = target_data_after_first_target_found['timestamp'] - target_data_after_first_target_found['timestamp'].iloc[idx_end_movement_after_target]
        
        #check
        
        # add a new column 'time_to_target_error' with the difference between the time to target estimation and the actual time to target
        target_data_after_first_target_found['time_to_target_error'] = target_data_after_first_target_found['time_to_target'] - target_data_after_first_target_found['estimated_target_time_to_impact']
        
        # compute the rmse of the time to target estimation
        time_to_target_rmse = np.sqrt(np.mean(target_data_after_first_target_found['time_to_target_error']**2))
        
        evaluation_data = {'cam_hand_position': cam_hand_position,
                           'trial_nb_frames': trial_nb_frames,
                           'trial_nb_achievable_task_detections': nb_achievable_task_detections,
                           
                           'idx_first_hand_found': idx_first_consecutive_hand_found,
                           'nb_hand_found': nb_hand_found,
                           'ratio_hand_found': ratio_hand_found,
                           'is_hand_estimation_continuous': is_hand_estimation_continuous,
                           
                           'idx_first_object_found': idx_first_consecutive_object_found,
                           'nb_object_found': nb_object_found,
                           'ratio_object_found': ratio_object_found,
                           'is_object_estimation_continuous': is_object_estimation_continuous,
                           
                           'trial_duration': trial_duration,
                           'first_hand_found': first_consecutive_hand_found_timestamp,
                           'first_object_found': first_consecutive_object_found_timestamp,                           
                           'first_target_found': first_target_found_timestamp, 
                           'first_grip_found': first_grip_found_timestamp, 
                           'target_found_delay': target_found_delay,
                            'grip_found_delay': grip_found_delay,
                           
                           'is_trial_valid' : is_trial_valid, 
                           'not_valid_reasons': not_valid_reasons,
                           'valid_reasons': valid_reasons,
                           
                           'nb_target_found': nb_target_found, 
                           'target_found_ratio': target_found_ratio, 
                           'ratio_target_switch': ratio_target_switch, 
                           'task_target_idenfication_successful': task_target_idenfication_successful,
                           
                           'nb_grips_found': nb_grips_found,
                           'grip_found_ratio': grip_found_ratio,
                           'ratio_grip_switch': ratio_grip_switch,
                           'task_grip_idenfication_successful': task_grip_idenfication_successful,
                           
                           'max_velocity': max_velocity,
                           'min_velocity': min_velocity,
                           'delta_velocity': delta_velocity,
                            'min_velocity_threshold': min_velocity_threshold,
                            'end_movement_timestamp': end_movement_timestamp,
                            'resting_time': resting_time,
                           'time_to_target_rmse': time_to_target_rmse
                           }
        
        return evaluation_data
        
    def analyse_old(self, experiment_analyser):
        if not self.was_replayed():
            print(f'Trial {self.label} not replayed. This trial cannot be analysed and will be skipped.')
            return
        print(f"Analysing trial {self.label}")
        if not os.path.exists(self.analysis_path):
            os.mkdir(self.analysis_path)
            
        device_id = experiment_analyser.get_device_id()
        main_data_file = [f for f in os.listdir(self.replay_path) if f.endswith('main.csv') and device_id in f][0]
        video_name = [f for f in os.listdir(self.pre_processing_path) if device_id in f and f.endswith(".avi") and 'movement' in f][0]
        video_path = os.path.join(self.pre_processing_path, video_name)
        print(f'video_path: {video_path}')

        
        hands_labels = []
        hands_trajs = [f for f in os.listdir(self.replay_path) if f.endswith('hand_traj.csv') and device_id in f]
        for hand_traj in hands_trajs:
            hand_label = hand_traj.split('_')[-3]
            # hand_label = hand_traj.split('_')[-3]+'_'+hand_traj.split('_')[-2]
            hands_labels.append(hand_label)
        print(f'hands_labels: {hands_labels}')
        #del duplicates
        hands_labels = set(hands_labels)
        print(f'hands_labels: {hands_labels}')
        
        objects_labels = []
        objects_trajs = [f for f in os.listdir(self.replay_path) if f.endswith('_traj.csv') and 'obj' in f and device_id in f]
        for object_traj in objects_trajs:
            object_label = object_traj.split('_')[-3]+'_'+object_traj.split('_')[-2]
            objects_labels.append(object_label)
        print(f'objects_labels: {objects_labels}')
        objects_labels = list(set(objects_labels))
        print(f'objects_labels: {objects_labels}')
        
        main_data_path = os.path.join(self.replay_path, main_data_file)
        print(f'main_data_path: {main_data_path}')
        main_data = pd.read_csv(main_data_path)
        
        hand_target_data = experiment_analyser.analyse(hands_labels, objects_labels, main_data, video_path=video_path)
        for hand, target_data in hand_target_data.items():
            target_data.to_csv(os.path.join(self.analysis_path, f"{self.label}_cam_{device_id}_{hand}_target_data.csv"), index=False)

    def save_replay_data(self, device_id):
        #write hands_data and objects_data to csv files
        for hand_id, hand_data in self.hands_data.items():
            hand_data = hand_data.drop_duplicates(subset=['Timestamps'])
            hand_data.to_csv(os.path.join(self.replay_path, f"{self.label}_cam_{device_id}_{hand_id}_traj.csv"), index=False)
            
        for object_id, object_data in self.objects_data.items():
            object_data = object_data.drop_duplicates(subset=['Timestamps'])
            object_data.to_csv(os.path.join(self.replay_path, f"{self.label}_cam_{device_id}_{object_id}_traj.csv"),index=False)
        
        # write main_data to csv file
        main_data = self.main_data.drop_duplicates(subset=['Timestamps'])
        main_data.to_csv(os.path.join(self.replay_path, f"{self.label}_cam_{device_id}_main.csv"), index=False)
        
        # write replay_monitoring to csv file
        self.replay_monitoring.to_csv(os.path.join(self.replay_path, f"{self.label}_cam_{device_id}_monitoring.csv"), index=False)
        
        # save the overlayed video
        overlayed_video_path = os.path.join(self.replay_path, f"{self.label}_cam_{device_id}_overlayed_video.avi")
        fps = 30
        res = self.saved_imgs[0].shape[:2][::-1]
        out = cv2.VideoWriter(overlayed_video_path, self.fourcc, fps, res)
        for img in self.saved_imgs:
            out.write(img)
        out.release()
        
        # save the replay data dictionary to a pkl file
        replay_data_path = os.path.join(self.replay_path, f"{self.label}_cam_{device_id}_replay_data.pkl")
        with open(replay_data_path, 'wb') as f:
            pickle.dump(self.replay_data_dict, f)
    
    def read_replay_data(self, device_id):
        # list all files from the trial folder, files only, that end with hand_traj.csv
        hand_files = [f for f in os.listdir(self.path) if os.path.isfile(os.path.join(self.path, f)) and device_id in f and f.endswith(f"hand_traj.csv")]
        for hand_file in hand_files:
            # get the hand id from the file name : between device_id and .csv
            hand_id = hand_file.split(device_id)[1].split(".csv")[0]
            hand_data = pd.read_csv(os.path.join(self.path, hand_file))
            self.hands_data[hand_id] = hand_data
            
        # list all files from the trial folder, files only, that end with object_traj.csv
        object_files = [f for f in os.listdir(self.path) if os.path.isfile(os.path.join(self.path, f)) and device_id in f and f.endswith(f"object_traj.csv")]
        for object_file in object_files:
            # get the object id from the file name : between device_id and .csv
            object_id = object_file.split(device_id)[1].split(".csv")[0]
            object_data = pd.read_csv(os.path.join(self.path, object_file))
            self.objects_data[object_id] = object_data
            
        self.main_data = pd.read_csv(os.path.join(self.path, f"{self.label}_cam_{device_id}_main.csv"))
    
    def analyse_data(self, experiment_analyser):
        
        device_id = experiment_analyser.get_device_id()
        self.read_replay_data(device_id)
        if self.hands_data is None or self.objects_data is None:
            print("No data to analyse")
            return
        else:
            print("Analysing data...")
            experiment_analyser.analyse(self.hands_data, self.objects_data)
            print("Data analysed")
    
    def check_and_make_dir(self):
        if os.path.exists(self.path):
            overwrite = messagebox.askyesno("Trial folder already exists", f"Trial folder {self.path} already exists. Do you want to overwrite it?")
            if overwrite:
                #copy the folder to a backup, adding a timestamp
                backup_path = os.path.join(self.path, f"bckp_{self.label}_combinations_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}")
                os.rename(self.path, backup_path)
                print(f"Trial folder backuped to {backup_path}")
            else:
                messagebox.showinfo("Trial skipped", f"Trial folder {self.path} skipped, going to next trial")
                return False
        os.makedirs(self.path)
        self.combination.to_csv(os.path.join(self.path, f"{self.label}_combinations.csv"), index=False)
        return True
        
        
    def get_instructions(self):
        print(f'Combination: \n {self.combination}')
        mov_type = self.combination[self.combination_header[self.movement_type_ind]]
        obj = self.combination[self.combination_header[self.obj_ind]]
        hand = self.combination[self.combination_header[self.hand_ind]]
        grip = self.combination[self.combination_header[self.grip_ind]]
        intro = f"{self.instructions['intro']} \n \n"
        t_obj = f"\t {self.instructions['object_intro']} {self.instructions[obj]} \n"
        t_hand = f"\t {self.instructions['hand_intro']} {self.instructions[hand]} \n"
        t_grip = f"\t {self.instructions['grip_intro']} {self.instructions[grip]} \n"
        t_mov = f"\n {self.instructions[mov_type]}"
        text = intro + t_obj + t_hand + t_grip + t_mov
        return text
        
    def get_instructions_colored(self):
        #TODO : add language selection
        print(f'Combination: {self.combination}')
        # mov_type = self.combination[self.combination_header[self.movement_type_ind]]
        # obj = self.combination[self.combination_header[self.obj_ind]]
        # hand = self.combination[self.combination_header[self.hand_ind]]
        # grip = self.combination[self.combination_header[self.grip_ind]]
        # intro = f" \n \n  \n \n  \n{self.instructions['intro']} \n \n \n"
        # t_obj = f"\t {self.instructions['object_intro']} "
        # t_hand = f"\t {self.instructions['hand_intro']} "
        # t_grip = f"\t {self.instructions['grip_intro']} "
        # t_obj_c = f"{self.instructions[obj]} \n \n"
        # t_hand_c = f"{self.instructions[hand]} \n \n"
        # t_grip_c = f"{self.instructions[grip]} \n \n"
        # t_mov_c = f"\t {self.instructions[mov_type]} \n \n"
        # text = [(intro, "intro"),
        #         (t_mov_c, "mov_type"),
        #         (t_hand,"normal"),
        #         (t_hand_c, "hand"),
        #         (t_grip, "normal"),
        #         (t_grip_c, "grip"),
        #         (t_obj, "normal"),
        #         (t_obj_c, "object")]
        text = [(f"INSTRUCTIONS", "intro")]

        return text
    def get_instructions_colored2(self):
        pass
        
class ProgressDisplay(ttk.Labelframe):
    def __init__(self, nb_items, items_label, parent = None, title='') -> None:
        super().__init__(parent, text=title)
        
        frame = ttk.Frame(self)
        frame.pack(padx=10, pady=10)
        self.current_item = ttk.Label(frame)
        self.current_item.grid(row=0, columnspan=2, sticky="w")
        self.pb = ttk.Progressbar(frame, orient="horizontal", length=200, mode="determinate")
        self.pb.grid(row=1, column=0, sticky="w")
        self.label = ttk.Label(frame)
        self.label.grid(row=1, column=1, sticky="w")
        
        self.reset(nb_items, items_label, 'Starting pre-processing')
        
    def set_current(self, current_item):
        self.current_item_text = current_item
        self.update()
        
    def increment(self):
        self.index += 1
        self.update()
        
    def update(self) -> None:
        if self.nb_items != 0:
            self.pb['value'] = int(100 * self.index / self.nb_items)
        else:
            self.pb['value'] = 100
        self.label['text'] = f"{self.index}/{self.nb_items} {self.items_label}"
        self.current_item['text'] = "Current : "+ self.current_item_text
    
    def reset(self, nb_items=None, items_label=None, current_item=None ):
        self.index = 0
        if nb_items is not None:
            self.nb_items = nb_items
        if items_label is not None:
            self.items_label = items_label
        if current_item is not None:
            self.current_item_text= current_item
        self.update()
        