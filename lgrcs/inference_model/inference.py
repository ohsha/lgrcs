import logging
import keyboard
import dlib
import cv2
import pickle
from threading import Thread, Event
import config_lgrcs as config
from inference_model.ptt import PTT
from inference_model.classify_record import classify_record


def inference():
    logger = logging.getLogger(config.PROJECT_NAME)
    logger.info('Initializing...')
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(config.PREDICTOR)

    with open(config.MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    logger.info('Warm-up camera...')
    streaming = cv2.VideoCapture(-1)

    ptt_event = Event()
    ptt_event.clear()

    button_thread = Thread(target=PTT, args=(ptt_event, streaming))
    button_thread.daemon = True
    button_thread.start()

    try:
        print('For starting press * s *')
        while(True):

            if(keyboard.is_pressed('s')):
                classify_record(streaming, detector, predictor, model, ptt_event)

    except Exception as e:
        logger.info(f"exception during process: {str(e)}")
        cv2.destroyAllWindows()
        streaming.release()

    finally:
        cv2.destroyAllWindows()
        streaming.release()
