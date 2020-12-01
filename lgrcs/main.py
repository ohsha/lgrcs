from trainer_model.trainer import trainer
from inference_model.inference import inference
from my_tools.logger import initialize_logger
import config_lgrcs as config
import argparse


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--mode", required=False, default="trainer",
        help="choose '-m trainer' for a new user train process, or '-m inference' for real-time model")
    args = ap.parse_args()

    initialize_logger(config.PROJECT_NAME, config.LOG_PATH, config.DEBUG_LEVEL)

    if args.mode == 'trainer':
        trainer()
    elif args.mode == 'inference':
        inference()
    else:
        print('unrecognized mode, please provide supported mode.')
        exit()


