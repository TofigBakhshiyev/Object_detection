import numpy as np 
import cv2

img = cv2.imread('Path/images/typewriter.jpg')
""" 
b = img[:,:,0]
print(b)
g = img[:,:,1]
r = img[:,:,2]

cv2.imshow('Blue', b)
cv2.imshow('Green', g)
cv2.imshow('Red', r) """

all_rows = open('Path/model/synset_words.txt').read().strip().split("\n")

classes = [r[r.find(' ') + 1:] for r in all_rows]

""" for (i, c) in enumerate(classes):
    if i == 4:
        break
    print(i, c) """

net = cv2.dnn.readNetFromCaffe('Path/model/bvlc_googlenet.prototxt',
'Path/model/bvlc_googlenet.caffemodel')

blob = cv2.dnn.blobFromImage(img, 1, (224,224))

net.setInput(blob)

output = net.forward()

#print(output)

idx = np.argsort(output[0])[::-1][:5]
print(idx)
""" for (i, id) in enumerate(idx):
    print('{}. {} ({}): Probability {:.3}%'.format(i+1, classes[id], id, output[0][id] * 100))

cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows() """

