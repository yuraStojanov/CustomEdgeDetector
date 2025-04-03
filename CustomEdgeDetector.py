import numpy as np
import cv2



# Функция рекурсивно проходит через все слабые границы, и если они соеденены с сильными, делает их самих сильными 
def find_weak_connection(i, j, new_edges, k = 100):
    # i, j - индексы сильной границы
    # new_edges - изображение с границами
    # k - макс. глубина рекурсии
    if k>0:
        for x in range(i-1,i+2):
            for y in range(j-1,j+2):
                if new_edges[x, y] == 100:
                    new_edges[i, j] = 255
                    find_weak_connection(x, y, new_edges, k-1)    
    return new_edges




def myCanny(image, low_threshold = 50, high_threshold=150, K = 0.48):

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Считываем размеры изображения
    [rows, columns] = np.shape(gray_image)

    # Ядра Собеля для осей X и Y
    # Множитель перед ядром стоит для компенсации толщины линий
    kernel_x = K*np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = K*np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Инициализация массивов для градиентов
    gradient_x = np.zeros_like(gray_image, dtype=np.float32)
    gradient_y = np.zeros_like(gray_image, dtype=np.float32)

    # Применение оператора Собеля
    for i in range(1, gray_image.shape[0] - 1):
        for j in range(1, gray_image.shape[1] - 1):
            gradient_x[i, j] = np.sum(gray_image[i-1:i+2, j-1:j+2] * kernel_x)
            gradient_y[i, j] = np.sum(gray_image[i-1:i+2, j-1:j+2] * kernel_y)


    # Освобождаем память
    del gray_image, kernel_x, kernel_y

    # Вычисление величины градиента
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    # Сохраняем градиент в формате uint8
    # Для более равномерного распределения гистограммы изображения использую данную функцию
    # Так детектор углов лучше выделяет контур изображения, но контур в некоторых местах может быть толщиной в несколько пикселей
    gradient_magnitude = np.clip(np.round(gradient_magnitude), 0, 255).astype(dtype=np.uint8)

    # Вычисление направления градиента
    gradient_direction = np.arctan2(gradient_y, gradient_x)

    # Освобождаем память
    del gradient_x, gradient_y,

    # Non-maximum suppression
    # Диапазон направлений градиента квантуем на 8 частей
    # Если величина градиента наибольшая среди соседей в направлении градиента, то сохраняем значение, иначе принимаем равным нулю  
    suppressed = np.zeros_like(gradient_magnitude)

    for i in range(1, rows - 1):
        for j in range(1, columns - 1):
            angle = gradient_direction[i, j] * 180. / np.pi
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180)  or (-180 <= angle <= -157.5): 
                neighbors = [gradient_magnitude[i, j-1], gradient_magnitude[i, j+1]]
            elif (22.5 <= angle < 67.5) or (-157.5 <= angle <= -112.5) :
                neighbors = [gradient_magnitude[i-1, j-1], gradient_magnitude[i+1, j+1]]
            elif (67.5 <= angle < 112.5) or ( -112.5<= angle < -67.5):
                neighbors = [gradient_magnitude[i-1, j], gradient_magnitude[i+1, j]]
            elif (112.5 <= angle < 157.5) or (-67.5 <= angle < -22.5):
                neighbors = [gradient_magnitude[i-1, j+1], gradient_magnitude[i+1, j-1]]
            if gradient_magnitude[i, j] >= np.max(neighbors):
                suppressed[i, j] = gradient_magnitude[i, j]   
    

    #Освобожаем память
    del gradient_direction, gradient_magnitude

    # Двойная пороговая фильтрация
    # Разделяем на "слабые" и "сильные" границы, в зависимости от величны градиента
    strong_edges = suppressed >= high_threshold
    weak_edges = (suppressed >= low_threshold) & (suppressed < high_threshold)

    #Освобожаем память
    del suppressed

    # Окончательное выделение границ. Если в диапазоне 3x3 рядом с слабой границей есть сильная, считаем ее сильной, если нет, то отбрасываем слабую
    # При этом, слабая граница, ставшая сильной, также может сделать сильной близкую к ней слабую границу. Для этого мы проходим по всем индексам сильных границ,
    # проверяя, есть ли в окрестностях слабые границы, и в случае наличия слабых границ делаем их сильными, проверяя теперь окрестности бывшей слабой границы

    new_edges = np.zeros([rows, columns])

    new_edges[strong_edges] = 255
    new_edges[weak_edges] = 100

    strong_x,strong_y = strong_edges.nonzero()
    for z in range(len(strong_x)):
        i = strong_x[z]
        j = strong_y[z] 
        new_edges = find_weak_connection(i, j, new_edges)

    #Очищаем от слабых границ
    new_edges[new_edges == 100] = 0
    return new_edges

image = cv2.imread('car.jpg')
new_edges = myCanny(image)

# Выполняем выделение границ по встроенному алгоритму cv2.Canny
cv2.imshow('new edges.jpg', new_edges)

low_threshold = 50
high_threshold = 150
canny_edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), low_threshold, high_threshold)

cv2.imshow('OpenCV Canny Edges', canny_edges)

# Оцениваем разницу между встроенным и собственным алгоритмом
# Для этого считаем число отличных пикселей, и делим на общее число пикселей
diff = abs(canny_edges - new_edges)
difference = round(np.count_nonzero(diff)/ (image.shape[0]*image.shape[1]),4)*100
cv2.putText(diff,'difference between images is '+str(difference) +'%',(50, 50),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255, 0, 0),1,cv2.LINE_AA)

cv2.imshow('difference between images', diff)
print('Разница между изображениями составляет',difference, '%')
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('difference_between_images.jpg', diff)
cv2.imwrite('OpenCV_Canny_Edges.jpg', canny_edges)
cv2.imwrite('new_edges.jpg', new_edges)
cv2.imwrite('bw.jpg',cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))





    
