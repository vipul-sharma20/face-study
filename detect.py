import sys
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

def main():
    count = 0
    sum_total = 0.0
    ratio_list = []
    for f in glob.glob('Color_Neutral_jpg/*.jpg'):

        img = cv2.imread(f)
        image, face_area, mouth_area = face_detect(img)
        if mouth_area > 0:
            print f, face_area/mouth_area
            sum_total += (face_area/mouth_area)
            ratio_list.append(face_area/mouth_area)

            count += 1
        if count == 150:
            break
        #cv2.imwrite('processed/'+str(count)+'.jpg', img)
    x = np.array(range(1, count+1))
    y = np.array(ratio_list)
    print len(x), len(y)
    fig, ax = plt.subplots()
    fit = np.polyfit(x, y, deg=1)
    ax.plot(x, fit[0] * x + fit[1], color='red')
    ax.scatter(x, y)

    plt.show()
    #plt.plot(range(1, count+1), ratio_list, 'ro')
    #plt.axis([0, count+1, 0, 50])

    #plt.show()
    print 'average: ', (sum_total/count)
    #cv2.imshow('img',img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


def mouth_detect(image):
    mouth_cascade = cv2.CascadeClassifier('mouth_cascade.xml')
    mouth = mouth_cascade.detectMultiScale(image)
    max_y = 0
    s = []
    mouth_area = 0
    for (mx, my, mw, mh) in mouth:
        if my > max_y:
            max_y = my
            s = [mx, my, mw, mh]
            mouth_area = mw * mh
    return s, mouth_area


def face_detect(img):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    face_area = 0
    mouth_area = 0
    roi_color = []
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        face_area = w * h

        s, mouth_area = mouth_detect(roi_gray)
        break

    if len(roi_color) > 0:
        """
        s[0] = mx
        s[1] = my
        s[2] = mw
        s[3] = mh
        """
        m = roi_color[s[1]:s[1]+s[3], s[0]:s[0]+s[2]]
        cv2.rectangle(roi_color, (s[0], s[1]), (s[0]+s[2], s[1]+s[3]), (0, 255, 255, 0), 2)
        return roi_color, m, face_area, mouth_area
    else:
        return img, face_area, mouth_area


def threshold(image):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.cvtColor(grey, (35,35), 0)
    _, thresh = cv2.threshold(blurred, 127, 255,
            cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return thresh

if __name__ == '__main__':
    main()

