import cv2
import keyboard
import time


def PTT(button_event, streaming):
    while (True):
        if (keyboard.is_pressed('space')):
            button_event.set()
            time.sleep(0.0005)

        elif(keyboard.is_pressed('e')):
            cv2.destroyAllWindows()
            streaming.release()

        else:
            button_event.clear()
            time.sleep(0.0005)
