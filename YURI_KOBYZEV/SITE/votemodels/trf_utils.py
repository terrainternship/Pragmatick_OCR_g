import cv2
import numpy as np
from imutils.perspective import four_point_transform
from PIL import Image
from scipy import stats
from scipy.stats import mode
#from skimage.io import imread
#import matplotlib.pyplot as plt
from skimage.feature import canny
from skimage.color import rgb2gray
from skimage.transform import hough_line, hough_line_peaks


# функция поворота
def rotate_image(image, angle): 
    (h, w) = image.shape[: 2]
    center = (w // 2, h // 2)
    angle = angle[0]
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    corrected = cv2.warpAffine(image, M, (w, h), flags = cv2.INTER_CUBIC, borderMode = cv2.BORDER_REPLICATE)
    return corrected


# выравнивание на основе преобразования Хафа
def deskew_hough(image): 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = canny(image)
# преобразвание Хафа
    tested_angles = np.deg2rad(np.arange(0.1, 180.0))
    h, theta, d = hough_line(edges, theta=tested_angles)
    accum, angles, dists = hough_line_peaks(h, theta, d)
    most_common_angle = mode(np.around(angles, decimals=2))[0]
    skew_angle = np.rad2deg(most_common_angle - np.pi/2)
    image = rotate_image(image, skew_angle)
    return image


# преобразование Фурье
def deskew_fourier(image): 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = canny(image)

    # преобразование Фурье
    f = np.fft.fft2(edges)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log(np.abs(fshift))

    r,c = magnitude_spectrum.shape
    magnitude_spectrum[int(r/2),int(c/2)] = 0

    f = []
    for ri in range(r):
        for ci in range(c):
           f.append([magnitude_spectrum[ri][ci], ci, ri])
    frequency_and_indexes = np.array(f)
    frequency_and_indexes = frequency_and_indexes[frequency_and_indexes[:,0].argsort()[::-1]][:30] # используем первые 30 частот
    slope, intercept, r_value, p_value, std_err = stats.linregress(frequency_and_indexes[:,1],frequency_and_indexes[:,2])
    rotation_angle = np.round(np.rad2deg(np.arctan(slope)-np.pi/2), decimals=2)
    image = rotate_image(image, rotation_angle)
    return image



def fp_transf_color(img):
    orig = img.copy()
    ratio=0.1
    screenCnt=None
    resize_image = cv2.resize(orig, None,fx=ratio,fy=ratio)
    gray = cv2.cvtColor(resize_image, cv2.COLOR_BGR2GRAY)
# Размываем для сглаживания и удаления шума
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
# Находим контуры по градиенту
    edged = cv2.Canny(blur,0, 255)
#    image=resize_image.copy()
    cnts, hierancy = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts[::-1]: # перебор в обратном порядке (с больших контуров)
      perimetr = cv2.arcLength(c, True)  # определяем длину закрытого контура
      approx=cv2.approxPolyDP(c, 0.01* perimetr, True)  # Аппроксимируем контур чтобы получить закрытый четырехугольник
      if len(approx)==4:  # Если обнаружен четырехугольник
        screenCnt = approx
        break
#    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
# Преобразование перспективы
    wraped = four_point_transform(orig, screenCnt.reshape(4, 2)/ratio)
# Показываем и сохраняем результат
#    wraped = cv2.cvtColor(wraped, cv2.COLOR_BGR2GRAY)
#    ref = cv2.threshold(wraped, 120, 255, cv2.THRESH_BINARY)[1]
#    cv2.imwrite('result.jpg', ref)
#    return ref
    return wraped



def fp_transf_bg(img):
    orig = img.copy()
    ratio=0.1
    screenCnt=None
    resize_image = cv2.resize(orig, None,fx=ratio,fy=ratio)
    gray = cv2.cvtColor(resize_image, cv2.COLOR_BGR2GRAY)
# Размываем для сглаживания и удаления шума
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
# Находим контуры по градиенту
    edged = cv2.Canny(blur,0, 255)

#    image=resize_image.copy()
    cnts, hierancy = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts[::-1]: # перебор в обратном порядке (с больших контуров)
      perimetr = cv2.arcLength(c, True)  # определяем длину закрытого контура
      approx=cv2.approxPolyDP(c, 0.01* perimetr, True)  # Аппроксимируем контур чтобы получить закрытый четырехугольник
      if len(approx)==4:  # Если обнаружен четырехугольник
        screenCnt = approx
        break
#    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
# Преобразование перспективы
    wraped = four_point_transform(orig, screenCnt.reshape(4, 2)/ratio)
# Показываем и сохраняем результат
    wraped = cv2.cvtColor(wraped, cv2.COLOR_BGR2GRAY)
    ref = cv2.threshold(wraped, 120, 255, cv2.THRESH_BINARY)[1]
#    cv2.imwrite('result.jpg', ref)
    return ref


def toImgOpenCV(imgPIL): # Conver imgPIL to imgOpenCV
    i = np.array(imgPIL) # After mapping from PIL to numpy : [R,G,B,A]
                         # numpy Image Channel system: [B,G,R,A]
    red = i[:,:,0].copy(); i[:,:,0] = i[:,:,2].copy(); i[:,:,2] = red;
    return i;

def toImgPIL(imgOpenCV): return Image.fromarray(cv2.cvtColor(imgOpenCV, cv2.COLOR_BGR2RGB));

