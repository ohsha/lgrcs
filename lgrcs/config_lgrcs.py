import os
from collections import OrderedDict

project_path = os.path.dirname(__file__)
PROJECT_NAME = os.path.split(project_path)[1]

GESTURES = ['close','kiss', 'open', 'smile']

# saving a clean photo (without lips overlay)
INCLUDE_CLEAN = True

PIXELS_SIZE = 400
NUM_ITER = 200
TEST_SIZE = 100

OUTPUT_PATH = os.path.join(project_path, 'output')
PREDICTOR = os.path.join(project_path,r'shape_predictor_68_face_landmarks.dat')

#################  For Inference Model ###################################
                                                                        ##
user_name = 'sharr'
USER_PATH = os.path.join(OUTPUT_PATH, user_name)
MODEL_PATH = os.path.join(USER_PATH, r'model.pkl')
LOG_PATH = os.path.join(USER_PATH, 'log/logs.log')
DEBUG_LEVEL = True
                                                                        ##
##########################################################################

LANDMARKS_INDEXES = {
    "Outer_lips": (48,60),
    "Inner_lips": (60, 68),
    "Total_points": (48, 68)}

COMMANDS_ = {0: 'close.mp3',
           1: 'kiss.mp3',
           2: 'open.mp3',
           3: 'smile.mp3'}

COMMANDS = {0: 'Hi',
           1: 'I love you',
           2: 'WOW',
           3: "I am happy"}
