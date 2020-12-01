import imutils
import cv2
import dlib
import os
import keyboard
from imutils import face_utils
import config_lgrcs as config
from my_tools import dev_tools as dvt


def instructions(gesture):
    print(f'Please look at the camera,'
          f' make sure your lips are marked with the red circles,'
          f' press the keyboard "space" and stay still with a **  {gesture}  ** mouth position, for {config.NUM_ITER}.')
    print('\n')


def collect_data(user):

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(config.PREDICTOR)
    streaming = cv2.VideoCapture(0)

    green_color = (0, 255, 1)
    red_color = (0, 1, 255)  # BGR

    try:
        for gesture in config.GESTURES:
            instructions(gesture)

            itr = 0
            collecting = True
            pressed = False
            while collecting:

                ret, frame = streaming.read()
                frame = imutils.resize(frame, width=400, height=400)
                overlay = frame.copy()

                if keyboard.is_pressed('space'):
                    pressed = True

                if itr < config.NUM_ITER and pressed:

                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray = dvt.image_enhancements(gray)

                    rects = detector(gray, 0)
                    if len(rects) > 0:

                        rect = rects[0]
                        landmarks = predictor(gray, rect)
                        landmarks = face_utils.shape_to_np(landmarks)

                        print(f'{gesture}  -  {itr}')
                        user.collector[gesture][itr] = {}

                        (i, j) = config.LANDMARKS_INDEXES['Outer_lips']
                        pt = config.LANDMARKS_INDEXES['Outer_lips'][0]
                        for coord in landmarks[i:j]:
                            user.collector[gesture][itr][pt] = coord
                            pt += 1
                            cv2.circle(overlay, tuple(coord), 1, green_color, -1)


                        (i, j) = config.LANDMARKS_INDEXES['Inner_lips']
                        pt = config.LANDMARKS_INDEXES['Inner_lips'][0]
                        for coord in landmarks[i:j]:
                            user.collector[gesture][itr][pt] = coord
                            pt += 1
                            cv2.circle(overlay, tuple(coord), 1, red_color, -1)

                        img_name = f'{gesture}_{itr}.jpg'
                        clean_save_path = os.path.join(user.gestures_path['clean'][gesture], img_name)
                        cv2.imwrite(clean_save_path, frame)
                        overlay_save_path = os.path.join(user.gestures_path['overlay'][gesture], img_name)
                        cv2.imwrite(overlay_save_path, overlay)

                        to_df = user.collector[gesture][itr]
                        to_df['label'] = gesture
                        to_df['iter'] = itr
                        user.df = user.df.append(to_df, ignore_index=True)

                        itr += 1

                if itr >= config.NUM_ITER:
                    collecting = False

                # cv2.imshow(gesture, overlay)
                # key = cv2.waitKey(1) & 0xFF
    except Exception as e:
        print(e)
        cv2.destroyAllWindows()
        streaming.release()
        # camera.close()
    finally:
        cv2.destroyAllWindows()
        streaming.release()
    return user
