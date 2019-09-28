import cv2 as cv
import time
from imutils.video import FPS
import numpy as np

# Load the model.
net = cv.dnn.readNet('CIFAR_Model_v2_RESNET.xml',
                     'CIFAR_Model_v2_RESNET.bin')
# Specify target device.
#net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
# Read an image.
camera = cv.VideoCapture(0)
#camera.resolution = (640, 480)
#camera.framerate = 16
#rawCapture = cv.PiRGBArray(camera, size=(640, 480))

time.sleep(0.1)
fps = FPS().start()

while True:
    ret, image = camera.read()
#    image = frame.array
    
# frame = cv.imread('/home/pi/Pictures/street_and_person.jpg')
# if frame is None:
#     raise Exception('Image not found!')

# Prepare input blob and perform an inference.
    blob = cv.dnn.blobFromImage(image, size=(640, 480), ddepth=cv.CV_8U)
    net.setInput(blob)
    out = net.forward()
    print(np.argmax(out, axis=0))
# Draw detected faces on the frame.
    for detection in out:
        # print(detection)
        confidence = float(detection[2])
        # print(confidence)
        xmin = int(detection[3] * image.shape[1])
        ymin = int(detection[4] * image.shape[0])
        xmax = int(detection[5] * image.shape[1])
        ymax = int(detection[6] * image.shape[0])
        if confidence > 0.5:
            print(confidence)
            cv.rectangle(image, (xmin, ymin), (xmax, ymax), color=(0, 255, 0))
            cv.putText(image, str(confidence), (xmin, ymin), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    cv.imshow("Frame", image)
    key = cv.waitKey(1) & 0xFF
    #rawCapture.truncate(0)
    fps.update()
            
    if key == ord("q"):
        break
        
fps.stop()
print('FPS:', fps.fps())
# Save the frame to an image file.
# cv.imwrite('out.png', frame)
