import sys
import cv2
import numpy as np

def main():
    #img = cv2.imread('pout1.jpg')
    image = face_detect(img)
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def mouth_detect(image):
    mouth_cascade = cv2.CascadeClassifier('mouth_cascade.xml')
    mouth = mouth_cascade.detectMultiScale(image)
    max_y = 0
    s = []
    for (mx, my, mw, mh) in mouth:
        if my > max_y:
            max_y = my
            s = [mx, my, mw, mh]
    return s


def face_detect(img):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        s = mouth_detect(roi_gray)

    cv2.rectangle(roi_color, (s[0], s[1]), (s[0]+s[2], s[1]+s[3]), (0, 255, 255, 0), 2)
    return roi_color


if __name__ == '__main__':
  main()
  