import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt


class FaceStudy(object):

    def main(self):
        """
        Driver function
        :return: None
        """
        count = 0
        sum_total = 0.0
        ratio_list = []
        for f in glob.glob('Color_Neutral_jpg/*.jpg'):

            img = cv2.imread(f)
            image, face_area, mouth_area = self.face_detect(img)
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

    def face_detect(self, image):
        """
        Face detection from data set
        :param image: jpg image for detection (color)
        :return: image, face area, mouth area (color)
        """
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        face_area = 0
        mouth_area = 0
        roi_color = []
        s = None
        for (x,y,w,h) in faces:
            cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = image[y:y+h, x:x+w]
            face_area = w * h

            s, mouth_area = self.mouth_detect(roi_gray)
            break

        if len(roi_color) > 0 and s:
            """
            s[0] = mx
            s[1] = my
            s[2] = mw
            s[3] = mh
            """
            mouth_crop = roi_color[s[1]:s[1]+s[3], s[0]:s[0]+s[2]]
            cv2.rectangle(roi_color, (s[0], s[1]), (s[0]+s[2], s[1]+s[3]), (0, 255, 255, 0), 2)
            gray = cv2.cvtColor(mouth_crop, cv2.COLOR_BGR2GRAY)
            thresh = self.threshold(gray)
            m, mouth_area = self.draw_contours(mouth_crop, thresh)
            self.detect_corners(gray, mouth_crop)
            return image, face_area, mouth_area
        else:
            return image, face_area, mouth_area

    def mouth_detect(self, image):
        """
        Mouth detection from face region
        :param image: grayscaled cropped image of face (gray)
        :return: coordinates of mouth, mouth area
        """
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

    def draw_contours(self, image, thresh):
        """
        Draw contours around mouth
        :param image: cropped mouth image (color)
        :param thresh: thresholded image to find contours
        :return: contoured image (color), mouth area
        """
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,
                    cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0
        if contours:
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > max_area:
                    max_area = area
            cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
        return image, max_area

    def threshold(self, image):
        """
        Thresholding of mouth region
        :param image: grayscaled cropped image of mouth (gray)
        :return: thresholded image
        """
        blurred = cv2.GaussianBlur(image, (35,35), 0)
        _, thresh = cv2.threshold(blurred, 127, 255,
                cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        return thresh

    def detect_corners(self, image, mouth_crop):
        """
        Shi-Tomasi corner detection in mouth region
        :param image: grayscaled cropped image of mouth (gray)
        :param mouth_crop: cropped image of mouth (color)
        :return: None
        """
        corners = cv2.goodFeaturesToTrack(image, 6, 0.01, 10)
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(mouth_crop, (x,y), 3, 255, -1)

if __name__ == '__main__':
    pass
