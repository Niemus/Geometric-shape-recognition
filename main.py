import cv2
import numpy as np

def read_image(filename):
    try:
        img = cv2.imread(filename)
        if img is None:
            raise FileNotFoundError("Ошибка: изображение не загружено! (Функция чтения)")
        return img
    except FileNotFoundError as e:
        print(str(e))
        return None

def detect_shape(image):
    if image is None:
        return "Ошибка: изображение не загружено! (Функция обнаружения)"

    # Изображение конвертируется в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Применение размытие Гаусса
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Применение детектора Кэнни
    edges = cv2.Canny(blurred, 50, 200)

    # Найдите контуры на изображении с обнаруженными краями
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        # Получите приблизительную многоугольную кривую для контура
        approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)

        # Проверьте количество сторон многоугольника
        if len(approx) == 3:
            return "На изображении треугольник!"
        elif len(approx) == 4:
            # Рассчитайте соотношение сторон, чтобы отличать квадрат от прямоугольника
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if 0.9 <= aspect_ratio <= 1.1:
                return "На изображении квадрат!"
            else:
                return "На изображении прямоугольник!"
        elif len(approx) >= 5 and len(approx) <= 8:
            # Подогните эллипс к контуру и проверьте его соотношение сторон
            (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
            aspect_ratio = MA / ma
            if 0.9 <= aspect_ratio <= 1.1:
                return "На изображении круг!"

    return "Форма не распознана :("

image_filenames = [r'C:\square.jpg', r'C:\circle.jpg', r'C:\triangle.jpg', r'C:\rectangle.jpg']
results = []

for filename in image_filenames:
    image = read_image(filename)
    result = detect_shape(image)
    results.append(result)

# Результаты
for res in results:
    print(res)
