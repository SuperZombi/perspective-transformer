# pip install opencv-python-headless
import cv2
import numpy as np
from scipy.spatial import distance as dist
from pathlib import Path

class Perspective:
    def __init__(self, file):
        self.file = Path(file)
        self.img = cv2.imread(file)

    def filename(self, text=""):
        return f"{self.file.with_suffix('')}{text}{self.file.suffix}"

    def preprocess(self):
        # Преобразование изображения в оттенки серого
        self.img_process = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        # Фильтрация
        # self.img_process = cv2.GaussianBlur(self.img_process, (5, 5), 0)
        self.img_process = cv2.medianBlur(self.img_process, 5)

        # Гамма-коррекция
        invGamma = 1.0 / 0.3
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        self.img_process = cv2.LUT(self.img_process, table)

        # Бинаризация
        ret, self.thresh = cv2.threshold(self.img_process, 40, 200, cv2.THRESH_BINARY)

        cv2.imwrite(self.filename("_binary"), self.thresh)
        return self.filename("_binary")

    def findContours(self):
        contours, hierarchy = cv2.findContours(self.thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Функция для нахождения самого большого прямоугольника среди контуров
        def biggestRectangle(contours):
            max_area = 0
            indexReturn = -1
            for index in range(len(contours)):
                i = contours[index]
                area = cv2.contourArea(i)
                if area > max_area:
                    max_area = area
                    indexReturn = index
            return indexReturn

        # Самая большая зона
        indexReturn = biggestRectangle(contours)
        self.contour = contours[indexReturn]

        cv2.imwrite(self.filename("_contour"), cv2.drawContours(self.img.copy(), [self.contour], 0, (0, 0, 255), 3))
        return self.filename("_contour")

    def findBox(self):
        # Округление контура до 4 точек
        epsilon = 0.1 * cv2.arcLength(self.contour, True)
        self.approximations = cv2.approxPolyDP(self.contour, epsilon, True)
        self.approximations = self.approximations.reshape(4,2)

        # rect = cv2.minAreaRect(hull)
        # box = cv2.boxPoints(rect)
        # box = np.intp(box)

        cv2.imwrite(self.filename("_box"), cv2.drawContours(self.img.copy(), [self.approximations], 0, (0, 255, 0), 2))
        return self.filename("_box")

    def apply_transform(self):
        def order_points(pts):
            xSorted = pts[np.argsort(pts[:, 0]), :]
            leftMost = xSorted[:2, :]
            rightMost = xSorted[2:, :]
            leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
            (tl, bl) = leftMost
            D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
            (br, tr) = rightMost[np.argsort(D)[::-1], :]
            return np.array([tl, tr, br, bl], dtype="float32")

        # Сортировка точек LeftTop, RightTop, BottomRight, BottomLeft
        box = np.array(self.approximations, dtype="int")
        src_pts = order_points(box)

        # Евклидово расстояние
        width = int(np.linalg.norm(src_pts[0] - src_pts[1]))
        height = int(np.linalg.norm(src_pts[0] - src_pts[3]))

        dst_pts = np.array([[0,0], [width,0], [width,height], [0,height]], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        result_img = cv2.warpPerspective(self.img, M, (width, height))

        cv2.imwrite(self.filename("_transformed"), result_img)
        return self.filename("_transformed")
