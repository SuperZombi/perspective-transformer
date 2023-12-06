import cv2
import numpy as np
from scipy.spatial import distance as dist

# Загрузка изображения
img = cv2.imread('2.jpg')

# Преобразование изображения в оттенки серого
img_process = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# img_process = cv2.GaussianBlur(img_process, (5, 5), 0)
img_process = cv2.medianBlur(img_process, 5)

# canny = cv2.Canny(blur, 75, 200)

# cv2.imwrite('blur.png', canny)

# Вычисление кривой гамма-коррекции
invGamma = 1.0 / 0.3
table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

# Применение гамма-коррекции с использованием таблицы преобразования
img_process = cv2.LUT(img_process, table)

# Применение бинаризации к откорректированному изображению
ret, thresh1 = cv2.threshold(img_process, 40, 200, cv2.THRESH_BINARY)

# cv2.imwrite('blur.png', thresh1)

# Нахождение контуров в бинаризованном изображении
contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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

# Получение индекса самого большого контура
indexReturn = biggestRectangle(contours)

clean_cnt = contours[indexReturn]

# cv2.imwrite('hull.png', cv2.drawContours(img, [contours[indexReturn]], 0, (0, 255, 0), 3))

# Округление контура до 4 точек
epsilon = 0.009 * cv2.arcLength(clean_cnt, True)
approximations = cv2.approxPolyDP(clean_cnt, epsilon, True)
approximations = approximations.reshape(4,2)

# rect = cv2.minAreaRect(hull)
# box = cv2.boxPoints(rect) # cv2.cv.BoxPoints(rect) for OpenCV <3.x
# box = np.intp(box)

# cv2.imwrite('test.png', cv2.drawContours(img, [approximations], 0, (0, 255, 0), 2))

def order_points(pts):
    xSorted = pts[np.argsort(pts[:, 0]), :]
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
    return np.array([tl, tr, br, bl], dtype="float32")

def perspective_transformation_2(img, box):
    """Perform perspective transformation for distorted license plates."""
    box = np.array(box, dtype="int")
    src_pts = order_points(box)

    # use Euclidean distance to get width & height
    width = int(np.linalg.norm(src_pts[0] - src_pts[1]))
    height = int(np.linalg.norm(src_pts[0] - src_pts[3]))

    dst_pts = np.array([[0,0], [width,0], [width,height], [0,height]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_img = cv2.warpPerspective(img, M, (width, height))

    return warped_img

result = perspective_transformation_2(img, approximations)
cv2.imwrite('final.png', result)
