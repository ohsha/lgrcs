import logging
import cv2
import time
import os
from imutils import face_utils
import config_lgrcs as config
from my_tools import dev_tools as dvt


logger = logging.getLogger(config.PROJECT_NAME)


def play_command(command):
    play = config.COMMANDS[command]
    os.system('espeak "{}"'.format(play))

    logger.info(f'gesture: {command} - {play}')
    time.sleep(2)


def classify_record(streaming, detector, predictor, model, ptt_event):
    start_time = cv2.getTickCount()
    logger.debug('[ record_classifying ]')
    previous_gesture = None
    count_gesture = 0

    while (streaming.isOpened()):
        ret, frame = streaming.read()

        if(ptt_event.isSet()):
            logger.debug('ptt_event was set')
            # waiting 10 sec for response.
            if((cv2.getTickCount() - start_time) / cv2.getTickFrequency() > 10):
                logger.info('Please, try again!')
                os.system('espeak "{}"'.format("Please, try again!"))
                time.sleep(1)
                start_time = cv2.getTickCount()

            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_img = dvt.image_enhancements(gray_img)

            rects = detector(gray_img, 0)
            if len(rects) > 0:
                logger.debug('rect founded')

                rect = rects[0]
                landmarks = predictor(gray_img, rect)
                landmarks = face_utils.shape_to_np(landmarks)
                logger.debug('landmarks predicted')

                (i, j) = config.LANDMARKS_INDEXES["Outer_lips"]
                for coord in landmarks[i : j]:
                    cv2.circle(frame, tuple(coord), 1,(0,0,255),-1)

                extracted_features = dvt.feature_extraction(landmarks).reshape(1,-1)
                pred_gesture = model.predict(extracted_features)
                pred_gesture = pred_gesture[0]
                logger.debug(f'gesture predicted: {pred_gesture}')

                if previous_gesture != pred_gesture:
                    count_gesture = 0
                    previous_gesture = pred_gesture
                    logger.debug('previous_gesture != pred_gesture')
                else:
                    count_gesture += 1
                    logger.debug('previous_gesture == pred_gesture')
                    if count_gesture < 3:
                        play_command(pred_gesture)
                        count_gesture = 0
                        previous_gesture = None

                        start_time = cv2.getTickCount()
        else:
            start_time = cv2.getTickCount()

        # cv2.imshow('frame',frame)
        # key = cv2.waitKey(1) & 0xFF
        # if key ==ord('q'):
        #     break
