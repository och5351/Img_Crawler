#from Crawler.GoogleImgCrawler import GoogleImgCrawler
import cv2
import os

class CrawlerController:
    GIC = None  # 구글 이미지 크롤러 컨트롤러

    search = None

    def __init__(self, URL, photoCount, search):
        #self.GIC = GoogleImgCrawler(URL, photoCount, search)
        self.search = search
        self.collecting_human()


    def collecting_human(self):

        face_cascade = cv2.CascadeClassifier('venv/lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
        eye_casecade = cv2.CascadeClassifier('venv/lib/site-packages/cv2/data/haarcascade_eye.xml')


        if self.search[:2] == '여자':

            maxCount = len(os.listdir(self.search))
            for i in range(1, 2):

                img = cv2.imread('C://Users/och5351/Desktop/github_och/Img_Crawler/female/female'+str(i)+'.png')
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    roi_gray = gray[y:y + h, x:x + w]
                    roi_color = img[y:y + h, x:x + w]
                    eyes = eye_casecade.detectMultiScale(roi_gray)
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

                cv2.imshow('Image view', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        else:
            maxCount = len(os.listdir(self.search))
            for i in range(1, maxCount + 1):
                cv2.imread(self.search + '/female' + str(i) + '.png')