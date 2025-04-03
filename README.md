## Задачи
Реализация детектора границ на языке Python, сравнение с реализацией opencv cv2.Sobel

## Ход работы
1. Применение оператора Собеля, описываемого ядрами 
2. Расчет величины и направления градиента для каждого пикселя 
3.  Реализация алгоритма non-maximum suppression снижения толщины границ В данной реализации выполнено квантование диапазона направлений градиента на 8 частей. 
4.  Применение двойной пороговой фильтрации Входными параметрами пороговой фильтрации приняты значения low_threshold = 50, high_threshold = 150. В фильтрации реализован рекурсивный перебор всех “слабых границ” соединённых с “сильными” самостоятельно, или через другую слабую границу.
## Данные
Для тестирования в работе использовалось изображение 
![Тестовое изображение](bw,jpg "Тестовое изображение")
Изображение полученное с помощью cv2.Sobel
![Изображение полученное с помощью cv2.Sobe](OpenCV_Canny_Edges,jpg "Изображение полученное с помощью cv2.Sobel")
Изображение полученное с помощью собственного детектора границ
![Изображение полученное с помощью собственного детектора границ](new_edges,jpg "Изображение полученное с помощью собственного детектора границ")
## Результат

Пиксельное несоответствие между изображениями представлено на рисунке. На рисунке выполнен вывод численного значения несоответствия, в данном случае он равен 2%.
![Пиксельная разница реализаций](difference_between_images,jpg "Пиксельная разница реализаций")

![Рис.1 - Одновременное сравнение всех конфигураций](1.png "Рис.1 - Одновременное сравнение всех конфигураций")



