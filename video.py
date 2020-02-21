import numpy as np 
import cv2

cap = cv2.VideoCapture('./out_dublin.mp4')

all_rows = open('./Ex_Files_Deep_Learning_OpenCV/Exercise Files/model/synset_words.txt').read().strip().split("\n")

classes = [r[r.find(' ') + 1:] for r in all_rows]

net = cv2.dnn.readNetFromCaffe('./Ex_Files_Deep_Learning_OpenCV/Exercise Files/model/bvlc_googlenet.prototxt',
'./Ex_Files_Deep_Learning_OpenCV/Exercise Files/model/bvlc_googlenet.caffemodel') 

if cap.isOpened() == False:
    print('Cannot open video file or stream') 
     

while True:
    ret, frame = cap.read()

    blob = cv2.dnn.blobFromImage(frame, 1, (224,224))

    net.setInput(blob)

    output = net.forward()

    r = 1
    for i in np.argsort(output[0])[::-1][:1]: 
        txt = ' "%s" probability "%.3f" ' %(classes[i], output[0][i] * 100)
        cv2.putText(frame, txt, (0, 20 + 40 * r), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
        r += 1 
    if ret == True:
        cv2.imshow('Frame', frame)

        if cv2.waitKey(25) & 0xFF == 27:
            break

    else:
        cap.release()
        cv2.destroyAllWindows()
        break

    