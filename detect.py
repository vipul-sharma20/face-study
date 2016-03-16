import cv2
import numpy as np

#img = cv2.imread('pout1.jpg')

mouth_cascade = cv2.CascadeClassifier('mouth_cascade.xml')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x,y,w,h) in faces:
      cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
      roi_gray = gray[y:y+h, x:x+w]
      roi_color = img[y:y+h, x:x+w]

      mouth = mouth_cascade.detectMultiScale(roi_gray)
      max_y = 0
      s = []
      for (mx, my, mw, mh) in mouth:
          if my > max_y:
              max_y = my
              s = [mx, my, mw, mh]
          cv2.circle(roi_color, (mx, my), 2, (255, 0, 0))
      print s[1]
      cv2.rectangle(roi_color, (s[0], s[1]), (s[0]+s[2], s[1]+s[3]), (0, 255, 255, 0), 2)

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
