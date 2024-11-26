#!/usr/bin/env python3

import argparse
import cv2
import tkinter as tk
from tkinter import ttk
import os
import pandas as pd
import threading

#TODO : MODIFY THIS FILE ACCORDING TO YOUR NEEDS

class ExperimentPreProcessor:
    """
    A class used to preprocess experiments by handling video and depth map data, 
    allowing for cutting, saving, and analyzing specific segments of the data.
    Attributes
    ----------
    name : str
        The name of the experiment preprocessor.
    saved_imgs : list
        A list to store the saved images.
    processing_window : tk.Toplevel
        The main window for the pre-processing GUI.
    image_label : ttk.Label
        Label to display images.
    participant_label : ttk.Label
        Label to display participant information.
    check_frame : ttk.LabelFrame
        Frame to hold check trial information.
    trial_label : ttk.Label
        Label to display trial information.
    combination_label : ttk.Label
        Label to display combination information.
    combination_respected : tk.BooleanVar
        Variable to track if the combination is respected.
    bcombination_respected : bool
        Boolean to track if the combination is respected.
    face_visible : tk.BooleanVar
        Variable to track if the face is visible.
    bface_visible : bool
        Boolean to track if the face is visible.
    cut_frame : ttk.LabelFrame
        Frame to hold cut trial information.
    start_var : tk.DoubleVar
        Variable to track the start frame index.
    end_var : tk.DoubleVar
        Variable to track the end frame index.
    return_mov_start_var : tk.DoubleVar
        Variable to track the return movement start frame index.
    duration_label : ttk.Label
        Label to display the duration of the trial.
    reach_start_label : ttk.Label
        Label to display the reaching start frame index.
    reach_start_trackbar : ttk.Scale
        Trackbar to select the reaching start frame index.
    reach_end_label : ttk.Label
        Label to display the reaching end frame index.
    reach_end_trackbar : ttk.Scale
        Trackbar to select the reaching end frame index.
    return_mov_start_label : ttk.Label
        Label to display the return movement start frame index.
    return_mov_start_trackbar : ttk.Scale
        Trackbar to select the return movement start frame index.
    play_movement_button : ttk.Button
        Button to play the movement segment.
    play_contact_button : ttk.Button
        Button to play the contact segment.
    play_return_button : ttk.Button
        Button to play the return segment.
    loop_var : tk.BooleanVar
        Variable to track if the video should loop.
    bloop : bool
        Boolean to track if the video should loop.
    rotate_var : tk.BooleanVar
        Variable to track if the video should be rotated.
    brotate : bool
        Boolean to track if the video should be rotated.
    cut_and_save_button : ttk.Button
        Button to cut and save the trial.
    next_button : ttk.Button
        Button to move to the next trial.
    fps : float
        Frames per second of the video.
    fourcc : int
        FourCC code for the video codec.
    resolution : tuple
        Resolution of the video.
    save_threads : list
        List to store the save threads.
    durations : dict
        Dictionary to store the durations of different segments.
    cv_window_name : str
        Name of the OpenCV window.
    nb_trials : int
        Number of trials.
    current_trial_index : int
        Index of the current trial.
    video_files : list
        List of video files.
    video_paths : list
        List of video paths.
    depthmap_paths : list
        List of depth map paths.
    timestamps_paths : list
        List of timestamps paths.
    timestamps : pd.DataFrame
        DataFrame to store the timestamps.
    nb_frames : int
        Number of frames in the video.
    current_frame_index : int
        Index of the current frame.
    go_on : bool
        Boolean to control the playback loop.
    stay : bool
        Boolean to control the main loop.
    Methods
    -------
    next_trial():
        Moves to the next trial.
    set_new_participant(participant_name, nb_trials):
        Sets a new participant and the number of trials.
    process_trial(folder_path=None, combination=None, destination_folder=None):
        Processes a trial by loading video, depth map, and timestamps data.
    get_duration():
        Calculates and updates the duration of different segments.
    skip_trial():
        Skips the current trial.
    pre_process(replays, name=None):
        Pre-processes the videos by setting up the GUI and loading data.
    cut_and_save():
        Starts a thread to cut and save the trial.
    cut_and_save_task():
        Cuts and saves the trial data into separate segments.
    play_movement():
        Plays the movement segment of the trial.
    play_contact():
        Plays the contact segment of the trial.
    play_return():
        Plays the return segment of the trial.
    play(start, end):
        Plays the video from the start frame to the end frame.
    loop():
        Toggles the loop playback option.
    rotate():
        Toggles the rotate video option.
    set_face_visible():
        Toggles the face visible option.
    set_combination_respected():
        Toggles the combination respected option.
    to_display(imgs, index=None):
        Displays the images in the OpenCV window.
    onChangeStart(trackbarValue):
        Updates the start frame index and displays the corresponding frame.
    onChangeEnd(trackbarValue):
        Updates the end frame index and displays the corresponding frame.
    onChangeReturnMovStart(trackbarValue):
        Updates the return movement start frame index and displays the corresponding frame.
    onChange(trackbarValue):
        Placeholder method for trackbar change.
    run():
        Starts the main loop of the GUI.
    stop():
        Stops the main loop and closes all windows.
    """
    def __init__(self, name = None) -> None:
        # Initialize saved images
        self.saved_imgs = None
        
        # Set the name of the experiment preprocessor
        if name is None:
            self.name = f'ExperimentPreProcessor'
        else:
            self.name = name
        
        # Set the position and size of the window
        self.x, self.y, self.w, self.h = 150, 20, 300, 400
        
        # Create the main processing window
        self.processing_window = tk.Toplevel()
        self.processing_window.title(f"{self.name} : Pre-processing")
        self.processing_window.geometry("1000x800")
        
        # Create a label to display images
        self.image_label = ttk.Label(self.processing_window)
        self.image_label.pack()
        
        # Create a label to display participant information
        self.participant_label = ttk.Label(self.processing_window, text=f"Participant : ")
        self.participant_label.pack(padx=10, pady=10)
        
        # Create a frame for checking trial information
        self.check_frame = ttk.LabelFrame(self.processing_window, text="Check trial")
        self.check_frame.pack()
        
        # Create a label to display trial information
        self.trial_label = ttk.Label(self.check_frame, text=f"Trial 0/0")
        self.trial_label.pack(padx=10, pady=10)
        
        # Create a label to display combination information
        self.combination_label = ttk.Label(self.check_frame, text=f"Combination {name}", justify=tk.CENTER)
        self.combination_label.pack(padx=10, pady=10)
        
        # Create a frame for check buttons
        check_buttons_frame = ttk.Frame(self.check_frame)
        check_buttons_frame.pack(padx=10, pady=10)
        
        # Create a check button to track if the combination is respected
        self.combination_respected = tk.BooleanVar()
        self.combination_respected.set(True)
        self.bcombination_respected = False
        self.combination_respected_button = ttk.Checkbutton(check_buttons_frame, text="Combination respected", variable=self.combination_respected, command=self.set_combination_respected)
        self.combination_respected_button.grid(row=0, column=0, padx=10, pady=10)
        
        # Create a check button to track if the face is visible
        self.face_visible = tk.BooleanVar()
        self.bface_visible = False
        self.face_visible_button = ttk.Checkbutton(check_buttons_frame, text="Face visible", variable=self.face_visible, command=self.set_face_visible)
        self.face_visible_button.grid(row=0, column=1, padx=10, pady=10)
        
        # Create a frame for cutting trial information
        self.cut_frame = ttk.LabelFrame(self.processing_window, text="Cut trial")
        self.cut_frame.pack(fill=tk.X, expand=True, padx=10, pady=10)
        
        # Initialize variables for start, end, and return movement start frame indices
        self.start_var = tk.DoubleVar()
        self.start_var.set(0)
        self.end_var = tk.DoubleVar()
        self.end_var.set(100)
        self.return_mov_start_var = tk.DoubleVar()
        self.return_mov_start_var.set(200)
        
        # Create a label to display the duration of the trial
        dur_lab = ttk.Label(self.cut_frame, text=f"Duration : ")
        dur_lab.pack(padx=10, pady=10)
        self.duration_label = ttk.Label(self.cut_frame, text=f"{self.end_var.get() - self.start_var.get()} frames")
        self.duration_label.pack(padx=10, pady=10)
        
        # Create frames for start and end trackbars
        start_frame = ttk.Frame(self.cut_frame)
        start_frame.pack(fill=tk.X, expand=True, padx=20)
        start_frame.columnconfigure(1, weight=1)
        
        end_frame = ttk.Frame(self.cut_frame)
        end_frame.pack(fill=tk.X, expand=True, padx=20)
        end_frame.columnconfigure(1, weight=1)
        
        # Create a trackbar and label for the reaching start frame index
        self.reach_start_label = ttk.Label(start_frame, text=f"Reaching start (frame index): {self.start_var.get()}")
        self.reach_start_label.grid(row=0, column=0, sticky=tk.W, padx=10, pady=10)
        self.reach_start_trackbar = ttk.Scale(start_frame, variable=self.start_var, from_=0, to=100, orient=tk.HORIZONTAL, command=self.onChangeStart)
        self.reach_start_trackbar.grid(row=0, column=1,  sticky='ew', padx=10, pady=10)
        
        # Create a trackbar and label for the reaching end frame index
        self.reach_end_label = ttk.Label(end_frame, text=f"Reaching end (frame index): {self.end_var.get()}")
        self.reach_end_label.grid(row=1, column=0, sticky=tk.W, padx=10, pady=10)       
        self.reach_end_trackbar = ttk.Scale(end_frame, variable=self.end_var, from_=0, to=100, orient=tk.HORIZONTAL, command=self.onChangeEnd)
        self.reach_end_trackbar.grid(row=1, column=1,columnspan=3, sticky='ew', padx=10, pady=10)
        
        # Create a trackbar and label for the return movement start frame index
        self.return_mov_start_label = ttk.Label(end_frame, text=f"Return start (frame index): {self.return_mov_start_var.get()}")
        self.return_mov_start_label.grid(row=2, column=0, sticky=tk.W, padx=10, pady=10)       
        self.return_mov_start_trackbar = ttk.Scale(end_frame, variable=self.return_mov_start_var, from_=0, to=100, orient=tk.HORIZONTAL, command=self.onChangeReturnMovStart)
        self.return_mov_start_trackbar.grid(row=2, column=1,columnspan=3, sticky='ew', padx=10, pady=10)
        
        # Create a frame for play buttons
        play_frame = ttk.Frame(self.cut_frame)
        play_frame.pack(padx=10, pady=10)
        
        # Create buttons to play different segments of the trial
        self.play_movement_button = ttk.Button(play_frame, text="Play movement", command=self.play_movement)
        self.play_movement_button.grid(row=0, column=0,  padx=10, pady=10)
        self.play_contact_button = ttk.Button(play_frame, text="Play contact", command=self.play_contact)
        self.play_contact_button.grid(row=0, column=1,  padx=10, pady=10)
        self.play_return_button = ttk.Button(play_frame, text="Play return", command=self.play_return)
        self.play_return_button.grid(row=0, column=2,  padx=10, pady=10)
        
        # Create a check button to toggle loop playback
        self.loop_var = tk.BooleanVar()
        self.bloop = False
        self.loop_button = ttk.Checkbutton(play_frame, text="Loop video play", command=self.loop, variable=self.loop_var)
        self.loop_button.grid(row=0, column=3, padx=10, pady=10)
        
        # Create a check button to toggle video rotation
        self.rotate_var = tk.BooleanVar()
        self.brotate = False
        self.rotate_button = ttk.Checkbutton(play_frame, text="Rotate video", command=self.rotate, variable=self.rotate_var)
        self.rotate_button.grid(row=0, column=4, padx=10, pady=10)
        
        # Create a button to cut and save the trial
        self.cut_and_save_button = ttk.Button(self.cut_frame, text="Cut and Save", command=self.cut_and_save)
        self.cut_and_save_button.pack(padx=10, pady=10)
        
        # Create a button to move to the next trial
        self.next_button = ttk.Button(self.processing_window, text="Next", command=self.next_trial)
        self.next_button.pack(padx=10, pady=10)
        self.next_button.config(state='disabled')
        
        # Set default values for video processing
        self.fps = 30.0
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.resolution = (1280, 720)
        self.save_threads = []
        self.processing_window.update()
        self.durations = 0.0
        
        # Create an OpenCV window for displaying video
        self.cv_window_name = 'Pre-processing'
        cv2.namedWindow(self.cv_window_name, cv2.WINDOW_NORMAL)
        cv2.createTrackbar('start', self.cv_window_name, 0, 100, self.onChange)
        
    def next_trial(self):
        # Stop the current trial and move to the next one
        self.stay = False
        cv2.waitKey(50)
        self.current_trial_index += 1
        self.next_button.config(state='disabled')
        # Deactivate check buttons
        self.processing_window.update()
    
    def set_new_participant(self, participant_name, nb_trials):
        # Set the participant name and number of trials
        self.participant_label.configure(text=f"Participant : {participant_name}")
        self.nb_trials = nb_trials
        self.current_trial_index = 0
    
    def process_trial(self, folder_path=None, combination=None, destination_folder=None):
        # Initialize trial settings
        self.combination_respected.set(True)
        self.bcombination_respected = True
        self.face_visible.set(False)
        self.bface_visible = False
        self.destination_folder = destination_folder
        folder = folder_path.split('/')[-1]
        
        # Update GUI elements with trial information
        self.check_frame.configure(text=f"Check trial {self.current_trial_index+1}/{self.nb_trials}")
        self.cut_frame.configure(text=f"Cut trial {self.current_trial_index+1}/{self.nb_trials}")
        self.trial_label.configure(text=folder)
        combi_txt = 'Combination : \n'
        for i in range(4):
            combi_txt += f'\n {combination.values[i]}'
        self.combination_label.configure(text=combi_txt)
        
        # List '.avi' files in the folder
        self.video_files = [f for f in os.listdir(folder_path) if f.endswith('.avi')]
        print(f"video_files : {self.video_files}")
        recording_paths = [os.path.join(folder_path, video_file).split('_video.avi')[0] for video_file in self.video_files]
        
        # Set paths for video, depth map, and timestamps
        self.video_paths = [path + '_video.avi' for path in recording_paths]
        self.depthmap_paths = [path + '_depth_map.gzip' for path in recording_paths]
        self.timestamps_paths = [path + '_timestamps.csv' for path in recording_paths]
        
        # Load timestamps
        self.timestamps = pd.read_pickle(self.timestamps_paths[0], compression='gzip')
        
        print(f"video_paths : {self.video_paths}")
        print(f"depthmap_paths : {self.depthmap_paths}")
        print(f"timestamps_paths : {self.timestamps_paths}")
        
        # Pre-process the trial
        self.pre_process(recording_paths)
        return self.bcombination_respected, not self.bface_visible, self.durations

    def get_duration(self):
        # Get the start, end, and return movement start frame indices
        self.start = int(self.start_var.get())
        self.end = int(self.end_var.get())
        self.return_mov_start = int(self.return_mov_start_var.get())
        
        # Calculate the duration of each segment using timestamps
        self.durations = {}
        self.durations['stand'] = self.timestamps['Timestamps'].iloc[self.start] - self.timestamps['Timestamps'].iloc[0]
        self.durations['movement'] = self.timestamps['Timestamps'].iloc[self.end] - self.timestamps['Timestamps'].iloc[self.start]
        self.durations['contact'] = self.timestamps['Timestamps'].iloc[self.return_mov_start] - self.timestamps['Timestamps'].iloc[self.end]
        self.durations['return'] = self.timestamps['Timestamps'].iloc[-1] - self.timestamps['Timestamps'].iloc[self.return_mov_start]
        self.durations['total'] = self.timestamps['Timestamps'].iloc[-1] - self.timestamps['Timestamps'].iloc[0]
        
        # Update the duration label with the calculated durations
        dur = ''
        i = 0
        for d, t in self.durations.items():
            if i < 3:
                dur += f"{d} : {t:.2f} s - "
            else:
                dur += f"{d} : {t:.2f} s"
            i += 1
        self.duration_label.configure(text=dur)
    
    def skip_trial(self):
        # Increment the current trial index to skip the trial
        self.current_trial_index += 1
    
    def pre_process(self, replays, name=None):
        # Load video files for the given replays
        self.videos = [cv2.VideoCapture(replay + '_video.avi') for replay in replays]
        
        # Get frame dimensions and number of frames
        frame_width = int(self.videos[0].get(3))
        frame_height = int(self.videos[0].get(4))
        print(f"frame_width : {frame_width}, frame_height : {frame_height}")
        nbf = int(self.videos[0].get(cv2.CAP_PROP_FRAME_COUNT))
        self.nb_frames = min(nbf, len(self.timestamps))
        
        print(f"nb_frames : {self.nb_frames} (min between video [{nbf}] and timestamps[{len(self.timestamps)}])")
        
        # Configure trackbars for frame selection
        self.reach_start_trackbar.configure(to=self.nb_frames-1)
        self.reach_end_trackbar.configure(to=self.nb_frames-1)
        self.return_mov_start_trackbar.configure(to=self.nb_frames-1)
        cv2.setTrackbarMax('start', self.cv_window_name, self.nb_frames-1)
        cv2.setTrackbarPos('start', self.cv_window_name, 0)
        
        # Set initial values for start, end, and return movement start frame indices
        self.start_var.set(0)
        self.end_var.set(int(self.nb_frames / 2))
        self.return_mov_start_var.set(int(self.nb_frames * 3 / 4))
        self.get_duration()
        self.current_frame_index = 0
        
        # Check if videos are opened successfully
        for vid in self.videos:
            if not vid.isOpened():
                print("Error reading video")
                exit()
        
        # Update trackbars and display initial frames
        self.onChangeEnd(int(self.nb_frames / 2))
        self.onChangeReturnMovStart(int(self.nb_frames * 3 / 4))
        self.onChangeStart(0)
        
        # Main loop to keep the GUI running
        self.stay = True
        while self.stay:
            cv2.waitKey(25)
            self.processing_window.update()
    
    def cut_and_save(self):
        # Get the start, end, and return movement start frame indices
        self.start = int(self.start_var.get())
        self.end = int(self.end_var.get())
        self.return_mov_start = int(self.return_mov_start_var.get())
        
        # Start a new thread to cut and save the trial
        print("start cut and save thread")
        sav_th = threading.Thread(target=self.cut_and_save_task)
        sav_th.start()
        self.save_threads.append(sav_th)
        
        # Enable the next button to move to the next trial
        self.next_button.config(state='normal')
        
    def cut_and_save_task(self):        
        print('beginning cut and save')
        
        # Copy paths to avoid modifying the original lists
        video_paths = self.video_paths.copy()   
        depthmap_paths = self.depthmap_paths.copy()
        timestamps_paths = self.timestamps_paths.copy()
        
        # Get start, end, and return movement start frame indices
        start = self.start
        end = self.end
        return_mov_start = self.return_mov_start
        
        # Get video properties
        brotate = self.brotate
        resolution = self.resolution
        fourcc = self.fourcc
        fps = self.fps
        nb_frames = self.nb_frames
        
        # Generate destination paths for saving the segments
        destination_paths = [os.path.join(self.destination_folder, video_file).split('_video.avi')[0] for video_file in self.video_files]
        
        # Define paths for different segments
        stand_video_paths = [path + '_video_stand.avi' for path in destination_paths]
        stand_depthmap_paths = [path + '_depth_map_stand.gzip' for path in destination_paths]
        stand_timestamps_paths = [path + '_timestamps_stand.gzip' for path in destination_paths]
        
        mov_video_paths = [path + '_video_movement.avi' for path in destination_paths]
        mov_depthmap_paths = [path + '_depth_map_movement.gzip' for path in destination_paths]
        mov_timestamps_paths = [path + '_timestamps_movement.gzip' for path in destination_paths]
        
        contact_video_paths = [path + '_video_contact.avi' for path in destination_paths]
        contact_depthmap_paths = [path + '_depth_map_contact.gzip' for path in destination_paths]
        contact_timestamps_paths = [path + '_timestamps_contact.gzip' for path in destination_paths]
        
        ret_video_paths = [path + '_video_return.avi' for path in destination_paths]
        ret_depthmap_paths = [path + '_depth_map_return.gzip' for path in destination_paths]
        ret_timestamps_paths = [path + '_timestamps_return.gzip' for path in destination_paths]
        
        print("begin video saving")
        for id, v_path in enumerate(video_paths):
            reader = cv2.VideoCapture(v_path)
            res = resolution
            stand_recorder = cv2.VideoWriter(stand_video_paths[id], fourcc, fps, res)
            movement_recorder = cv2.VideoWriter(mov_video_paths[id], fourcc, fps, res)
            contact_recorder = cv2.VideoWriter(contact_video_paths[id], fourcc, fps, res)
            return_recorder = cv2.VideoWriter(ret_video_paths[id], fourcc, fps, res)
            frame_index = 0
            
            # Read and save frames to corresponding segments
            while reader.isOpened():
                err, img = reader.read()
                if frame_index < start:
                    stand_recorder.write(img)                
                elif frame_index >= start and frame_index <= end:
                    movement_recorder.write(img)
                elif frame_index > end and frame_index <= return_mov_start:
                    contact_recorder.write(img)
                elif frame_index > return_mov_start:
                    return_recorder.write(img)
                if frame_index == nb_frames - 1:                     
                    break
                frame_index += 1
            
            # Release video writers and reader
            stand_recorder.release()
            movement_recorder.release()
            contact_recorder.release()
            return_recorder.release()
            reader.release()
            print(f'video {id} saved')
        
        print("begin depthmap saving")
        for id, d_path in enumerate(depthmap_paths):
            df = pd.read_pickle(d_path, compression='gzip')
            
            # Save depth maps for different segments
            if start > 0:
                df_stand = df[:start]
                df_stand.loc[:, 'Timestamps'] = df_stand['Timestamps'] - df_stand['Timestamps'].iloc[0]
                df_stand.to_pickle(stand_depthmap_paths[id], compression='gzip')
            else:
                df_stand = pd.DataFrame(columns=df.columns)
            
            df_mov = df[start:end]
            df_mov.loc[:, 'Timestamps'] = df_mov['Timestamps'] - df_mov['Timestamps'].iloc[0]
            df_mov.to_pickle(mov_depthmap_paths[id], compression='gzip')
            
            df_con = df[end:return_mov_start]
            df_con.loc[:, 'Timestamps'] = df_con['Timestamps'] - df_con['Timestamps'].iloc[0]
            df_con.to_pickle(contact_depthmap_paths[id], compression='gzip')
            
            if return_mov_start < nb_frames - 1:
                df_ret = df[return_mov_start:]
                df_ret.loc[:, 'Timestamps'] = df_ret['Timestamps'] - df_ret['Timestamps'].iloc[0]
                df_ret.to_pickle(ret_depthmap_paths[id], compression='gzip')
            else:
                df_ret = pd.DataFrame(columns=df.columns)
            
            print(f'depthmap {id} saved')
        
        print("begin timestamps saving")
        for id, t_path in enumerate(timestamps_paths):
            df = pd.read_pickle(t_path, compression='gzip')
            
            # Save timestamps for different segments
            if start > 0:
                df_stand = df[:start]
                df_stand.loc[:, 'Timestamps'] = df_stand['Timestamps'] - df_stand['Timestamps'].iloc[0]
                df_stand.to_pickle(stand_timestamps_paths[id], compression='gzip')
            else:
                df_stand = pd.DataFrame(columns[df.columns])
            
            df_mov = df[start:end]
            df_mov.loc[:, 'Timestamps'] = df_mov['Timestamps'] - df_mov['Timestamps'].iloc[0]
            df_mov.to_pickle(mov_timestamps_paths[id], compression='gzip')
            
            df_con = df[end:return_mov_start]
            df_con.loc[:, 'Timestamps'] = df_con['Timestamps'] - df_con['Timestamps'].iloc[0]
            df_con.to_pickle(contact_timestamps_paths[id], compression='gzip')
            
            if return_mov_start < nb_frames - 1:
                df_ret = df[return_mov_start:]
                df_ret.loc[:, 'Timestamps'] = df_ret['Timestamps'] - df_ret['Timestamps'].iloc[0]
                df_ret.to_pickle(ret_timestamps_paths[id], compression='gzip')
            else:
                df_ret = pd.DataFrame(columns=df.columns)
            
            print(f'timestamps {id} saved')
            movement_time = df_mov['Timestamps'].iloc[-1] - df_mov['Timestamps'].iloc[0]
        
        print("end cut and save")
        return movement_time
    
    def play_movement(self):
        # Get the start and end frame indices for the movement segment
        self.start = int(self.start_var.get())
        self.end = int(self.end_var.get())
        # Play the movement segment
        self.play(self.start, self.end)
    
    def play_contact(self):
        # Get the end and return movement start frame indices for the contact segment
        self.end = int(self.end_var.get())
        self.return_mov_start = int(self.return_mov_start_var.get())
        # Play the contact segment
        self.play(self.end, self.return_mov_start)
        
    def play_return(self):
        # Get the return movement start frame index for the return segment
        self.return_mov_start = int(self.return_mov_start_var.get())
        # Play the return segment
        self.play(self.return_mov_start, self.nb_frames-1)
        
    def play(self, start, end):
        # Set the flag to continue playback
        self.go_on = True
        # Loop through the frames from start to end
        for frame_index in range(start, end):
            print(f'play, frame_index : {frame_index}')
            imgs = []
            # Read and display frames from all videos
            for vid in self.videos:
                vid.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))
                err, img = vid.read()
                imgs.append(img)
            self.saved_imgs = imgs
            self.to_display(imgs, frame_index)
            # Break the loop if the flag is set to False
            if not self.go_on:
                break
        # If looping is enabled, replay the segment
        if self.bloop and self.go_on:
            self.play(start, end)
    
    def loop(self):
        # Toggle the loop playback option
        self.bloop = not self.bloop
        print(f'loop : {self.bloop}')
        
    def rotate(self):
        # Toggle the rotate video option
        self.brotate = not self.brotate
        print(f'rotate : {self.brotate}')
        # Redisplay the saved images with the new rotation setting
        self.to_display(self.saved_imgs)
    
    def set_face_visible(self):
        # Toggle the face visible option
        self.bface_visible = not self.bface_visible
        print(f'face visible : {self.bface_visible}')
    
    def set_combination_respected(self):
        # Toggle the combination respected option
        self.bcombination_respected = not self.bcombination_respected
        print(f'combination respected : {self.bcombination_respected}')
        
    def to_display(self, imgs, index = None):
        nimgs = []
        for img in imgs:
            if img is None:
                return
            # Resize the image if needed
            # nimg = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_AREA)
            nimg = img
            # Rotate the image if the rotate option is enabled
            if self.brotate:
                nimg = cv2.rotate(nimg, cv2.ROTATE_90_CLOCKWISE)
            nimgs.append(nimg)
        # Concatenate images horizontally or vertically based on the rotate option
        if self.brotate:
            cimg = cv2.hconcat(nimgs)
        else:
            cimg = cv2.vconcat(nimgs)
        # Update the trackbar position if an index is provided
        if index is not None:            
            cv2.setTrackbarPos('start', self.cv_window_name, index)
        # Display the concatenated image
        cv2.imshow(self.cv_window_name, cimg)
        # Wait for a key press and stop playback if 'Esc' is pressed
        k = cv2.waitKey(25)
        if k == 27:
            self.go_on = False
        
    def onChangeStart(self, trackbarValue):
        print(f'onChangeStart, trackbarval : {trackbarValue}')
        st = float(trackbarValue)
        et = self.end_var.get()
        rt = self.return_mov_start_trackbar.get()
        
        # Ensure the start value is within valid range
        if st >= self.nb_frames - 1:
            self.reach_start_trackbar.set(self.nb_frames - 2)
            st = self.nb_frames - 2
        
        # Adjust the end trackbar if necessary
        if st >= et:
            self.reach_end_trackbar.set(st + 1)
            
        ind = int(float(trackbarValue))
        imgs = []
        # Read and display frames from all videos at the new start position
        for vid in self.videos:
            vid.set(cv2.CAP_PROP_POS_FRAMES, float(trackbarValue))
            err, img = vid.read()
            imgs.append(img)
        self.to_display(imgs, ind)
        print(f'start_var bef : {self.start_var.get()}')
        self.start_var.set(ind)
        print(f'start_var aft: {self.start_var.get()}')
        # Adjust spacing for the label based on the index value
        if ind >= 100:
            space = ''
        elif ind >= 10:
            space = ' '
        else:
            space = '  '
        self.reach_start_label.configure(text=f"Reaching start : {space}{ind}")
        self.saved_imgs = img
        self.get_duration()
        
    def onChangeEnd(self, trackbarValue):
        
        et = float(trackbarValue)
        st = self.start_var.get()
        rt = self.return_mov_start_trackbar.get()
        
        # Ensure the end value is within valid range
        if et <= 1:
            self.reach_end_trackbar.set(2)
            self.return_mov_start_trackbar.set(2)
            et = 2
            
        # Adjust the start trackbar if necessary
        if st >= et:
            self.reach_start_trackbar.set(et - 1)
        
        # Adjust the return movement start trackbar if necessary
        if et >= rt:
            self.return_mov_start_trackbar.set(et + 1)
            
        ind = int(float(trackbarValue))
        imgs = []
        # Read and display frames from all videos at the new end position
        for vid in self.videos:
            vid.set(cv2.CAP_PROP_POS_FRAMES, float(trackbarValue))
            err, img = vid.read()
            imgs.append(img)
        self.to_display(imgs, ind)
        
        self.end_var.set(ind)
        # Adjust spacing for the label based on the index value
        if ind >= 100:
            space = '  '
        elif ind >= 10:
            space = '   '
        else:
            space = '    '
        self.reach_end_label.configure(text=f"Reaching end : {space}{ind}")
        self.saved_imgs = img
        self.get_duration()
        
    def onChangeReturnMovStart(self, trackbarValue):
        rt = float(trackbarValue)
        et = self.end_var.get()
        st = self.start_var.get()
        
        # Ensure the return movement start value is within valid range
        if rt <= 1:
            self.reach_end_trackbar.set(2)
            self.reach_start_trackbar.set(2)
            rt = 2
            
        # Adjust the end trackbar if necessary
        if et >= rt:
            self.reach_end_trackbar.set(rt - 1)
            
        ind = int(float(trackbarValue))
        imgs = []
        # Read and display frames from all videos at the new return movement start position
        for vid in self.videos:
            vid.set(cv2.CAP_PROP_POS_FRAMES, float(trackbarValue))
            err, img = vid.read()
            imgs.append(img)
        self.to_display(imgs, ind)
        
        self.return_mov_start_var.set(ind)
        # Adjust spacing for the label based on the index value
        if ind >= 100:
            space = '  '
        elif ind >= 10:
            space = '   '
        else:
            space = '    '
        self.return_mov_start_label.configure(text=f"Return start : {space}{ind}")
        self.saved_imgs = img
        self.get_duration()
    
    def onChange(self, trackbarValue):
        pass
    
    def run(self):
        # Start the main loop of the GUI
        self.processing_window.mainloop()
    
    def stop(self):
        # Destroy all OpenCV windows and stop the main loop
        cv2.destroyAllWindows()
        self.stay = False
        # Wait for all save threads to finish
        for th in self.save_threads:
            th.join()
        # Destroy the processing window
        self.processing_window.destroy()

        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-hd', '--hand_detection', choices=['mediapipe', 'depthai', 'hybridOAKMediapipe'],
                        default = 'hybridOAKMediapipe', help="Hand pose reconstruction solution")
    parser.add_argument('-od', '--object_detection', choices=['cosypose, megapose'],
                        default = 'cosypose', help="Object pose reconstruction detection")
    args = vars(parser.parse_args())

    # if args.hand_detection == 'mediapipe':
    #     import mediapipe as mp
    # else:
    #     import depthai as dai
    
    # if args.object_detection == 'cosypose':
    #     import cosypose
    i_grip = ExperimentPreProcessor('test')
    i_grip.run()