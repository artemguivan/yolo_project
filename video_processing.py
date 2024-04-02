import cv2 
import copy
import datetime
from tools.preprocessing import get_first_frame_info, get_fps
from ultralytics import YOLO
import pandas as pd
import moviepy.editor as mp
import supervision as sv
import numpy as np
import PIL
import time
import csv

GREEN = (0, 255, 0)
model = YOLO(r"weights\best_pupil_2_4_24.onnx")
path_to_processed_video = r"out_5sec.mp4" # сюда вставляем путь, в котором будет находиться обработанное видео
path = r"saved_video\test5sec.mp4" # сюда вставляем путь к видео, которое хотим обработать 
center_x_list = []
center_y_list = []
frames = []
# чтобы в отдельном окне отображать обработку видео поменяйте значение в show_flag на True
show_flag = False

def create_video_writer(path, output_filename) -> None:
    """
    Функция принимает путь к видео и выдаёт объект для создания видео из кадров

    path: путь к видео
    output_filename: путь к месту, в котором мы хотим сохранить видео
    
    """
     
    video = mp.VideoFileClip(path)
    first_frame = video.get_frame(0)
    frame_width, frame_height = PIL.Image.fromarray(first_frame).size
    fps = get_fps(path)

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    writer = cv2.VideoWriter(output_filename, fourcc, fps,
                             (frame_width, frame_height))

    return writer

writer = create_video_writer(path, path_to_processed_video)
start_time = time.time()
video_cap = cv2.VideoCapture(path)

def main():
    if not video_cap.isOpened():
        print("Ошибка открыьтия файла")
    else:
        fps = video_cap.get(cv2.CAP_PROP_FPS)
        print('Кадров в секунду:', fps, 'FPS')

        frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print('Количество кадров:', frame_count)

    frame_number = 0
    while video_cap.isOpened():
        
        ret, frame = video_cap.read()
        if ret:
            frame_number +=1
            predict = model(frame, task="detect", device="cpu")[0]
            

            if int(sv.Detections.from_ultralytics(predict).xyxy.size) == int(0):
                print("Не найдено зрачка на фото")
                writer.write(frame)
                center_x_list.append(0)
                center_y_list.append(0)
                frames.append(frame_number)
                if show_flag == True:
                    cv2.imshow("КАДР", frame)
            else:
                coordinates = copy.deepcopy(sv.Detections.from_ultralytics(predict).xyxy)
                x1, y1, x2, y2 = int(coordinates[0][0]), int(coordinates[0][1]), int(coordinates[0][2]), int(coordinates[0][3])
                cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 2)
                image_np = cv2.cvtColor(np.array(PIL.Image.fromarray(predict.orig_img)), cv2.COLOR_RGB2BGR)

            # считаем центр зрачка
                center_x = round(((coordinates[0][2]+coordinates[0][0]) / 2), 6)
                center_y = round(((coordinates[0][3]+coordinates[0][1]) / 2), 6)
                center_x_list.append(center_x)
                center_y_list.append(center_y)
                frames.append(frame_number)
    
            # отрисовываем прямоугольник
                cv2.rectangle(image_np, (x1, y1), (x2, y2), GREEN, 2)
                
            # отрисовываем центр зрачка
                cv2.circle(image_np, (int(center_x), int(center_y)), radius=3, color=GREEN, thickness=-1)
                writer.write(image_np)
                if show_flag == True:
                    cv2.imshow("КАДР", image_np)

            key = cv2.waitKey(20)
            if key == ord('q'):
                print("Стоп")
                break
            else:
                print("Нажата кнопка:", key)
        else:
            print("Взятие видео закончено")
            break
    
    df = pd.DataFrame({'center_x': center_x_list, 'center_y': center_y_list, "frame": frames})
    excel_filename = path_to_processed_video.replace(".mp4", "_frames.xlsx")
    df.to_excel(excel_filename, index=False)
    end_time = time.time()
    execution_time = end_time - start_time
    print("Обработка заняла:", execution_time, "секунд")

    video_cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
