import config_lgrcs as config
import pandas as pd
import numpy as np
import csv
import os


class User:

    def __init__(self, name):

        self.name = str(name)
        self.parent_dir = os.path.join(config.OUTPUT_PATH, name)
        self.model_path = os.path.join(self.parent_dir, 'model.pkl')
        self.csv_path = os.path.join(self.parent_dir, 'coords.csv')
        self.pickle_path = os.path.join(self.parent_dir, 'coords.pkl')
        self.photo_dir = os.path.join(self.parent_dir, 'photos')
        self.clean_photo = os.path.join(self.photo_dir, 'clean')
        self.overlay_photo = os.path.join(self.photo_dir, 'overlay')
        self.log_dir = os.path.join(self.parent_dir, 'log')
        self.gestures_path = {'overlay': {}}
        self.collector = {}

        for gesture in config.GESTURES:
            self.collector[gesture] = {}

        assert not os.path.exists(self.parent_dir), 'User already exist, please provide a new user name.'
        self._create_dir()
        self.df = self._create_data_frame()

    def _create_dir(self):
        if config.INCLUDE_CLEAN == True:
            self.gestures_path['clean'] = {}

        directories = [self.parent_dir, self.photo_dir, self.clean_photo, self.overlay_photo]
        for dir in directories:
            os.makedirs(dir)

        for gesture in config.GESTURES:
            overlay_path = os.path.join(self.overlay_photo, gesture)
            self.gestures_path['overlay'][gesture] = overlay_path
            os.makedirs(overlay_path)

            if config.INCLUDE_CLEAN == True:
                clean_path = os.path.join(self.clean_photo, gesture)
                self.gestures_path['clean'][gesture] = clean_path
                os.makedirs(clean_path)

    def _create_data_frame(self):

        init, end = config.LANDMARKS_INDEXES['Total_points']
        columns = [i for i in range(init, end)]
        columns.append('label')
        columns.append('iter')

        return pd.DataFrame(columns=columns, dtype=np.int32)

    def export_collected_data(self, include_pickle=True):

        self.df.to_csv(self.csv_path, index=False)
        if include_pickle:
            self.df.to_pickle(self.pickle_path)
        print('[INFO] collected-data was exported.')

