## Tools

### Preprocessing
```
timer - декоратор, который позволяет сколько времени функция выполняла свою задачу

cut_video - функция, позволяющая вырезать часть видео и сохранить его.
Аргументы:
path: путь к видео, которое Вы хотите обрезать
start: секунда с которой будет обрезано видео 
end: секунда до которой будет обрезано видео
savepath: место, в которое сохраняем обработанное видео

get_fps - Функция принимает путь к видео и выдаёт количество кадров в секунду(fps) в целых числах.
Аргументы:
path: путь к видео

get_first_frame_info - Функция принимает путь к видео и выдаёт размер первого кадра.
Аргументы:
path: путь к видео

cut_video_on_frames - Функция принимает путь к видео и каждый второй кадр добавляет в директорию, которая была указана в аргументах
Аргументы:
path: путь к видео, которое Вы хотите обрезать
path_to_save: место, в которое сохраняем обработанное видео
```

### Video_processing

```
create_video_writer - Функция принимает путь к видео и выдаёт объект для создания видео из кадров
Аргументы:
path: путь к видео
output_filename: путь к месту, в котором мы хотим сохранить видео

main -функция, котораязапускает основную часть программы(запускает видео и начинает его обработку, записывает координаты зрачка в файл)
```

## Разработка 

```
Для реализации была выбрала модель YOLOv8(формат - .pt), модель была обучена на выборке из 1060 изображений для тренировки,
100 изображений для валидации и теста(с глинтами, узким зрачком и т.д), далее была сделана квантизация и перевод модели
в формат onnx для того, чтобы уменьшить время инференса модели.Итоговые размеры изображений для обучения - 896*672.

Также для решения задачи пробовали другие алгоритмы глубокого обучения: YOLOv5, DETR, но они уступали
по скорости и по точночти предсказания. Также возможен перевод модели в формат TensorRT, но время инференса
модели уменьшится по большей части только для GPU.

На момент 29.02.2024 время обработки видео длинной m занимает примерно 3*m.

```