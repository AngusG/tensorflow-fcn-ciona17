import cv2
# More info on OpenCV Codecs
# http://docs.opencv.org/3.1.0/dd/d43/tutorial_py_video_display.html

def run():
    """Runs camera test"""
    cap = cv2.VideoCapture(0)

    ret = cap.set(3,320) 
    ret = cap.set(4,240)
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            cv2.imshow('frame',frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    # Release everything if job is finished
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    run()