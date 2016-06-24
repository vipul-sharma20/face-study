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
        cv2.imwrite('processed_corners/'+str(count)+'.jpg', img)
    x = np.array(range(1, count+1))
    y = np.array(ratio_list)
    print len(x), len(y)
    fig, ax = plt.subplots()
    fit = np.polyfit(x, y, deg=1)
    ax.plot(x, fit[0] * x + fit[1], color='red')
    ax.scatter(x, y)

    plt.show()
    print 'average: ', (sum_total/count)


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
        gray = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
        thresh = threshold(gray)
        m, mouth_area = draw_contours(m, thresh)
        detect_corners(gray, m)
        return img, face_area, mouth_area
    else:
        return img, face_area, mouth_area


def draw_contours(img, thresh):
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        max_area = 0
        c = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                c = contour
        cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    return img, max_area


def threshold(image):
    blurred = cv2.GaussianBlur(image, (35,35), 0)
    _, thresh = cv2.threshold(blurred, 127, 255,
            cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return thresh


def detect_corners(image, m):
    corners = cv2.goodFeaturesToTrack(image, 6, 0.01, 10)
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(m, (x,y), 3, 255, -1)

if __name__ == '__main__':
    main()

