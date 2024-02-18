import copy
import io
import numpy as np
import PIL 
import supervision as sv
import cv2
import streamlit as st
from ultralytics import YOLO
import pandas as pd
import moviepy.editor as mp

import time
import tempfile

class TextShower:
    def __init__(self, duration: float):

            self.duration = duration

    def show_text(self):  
        if str(round(self.duration*3, 2))[-1] == "1":
            uploaded_text = st.text(f"Видео будет обработано примерно за : {round(self.duration*3, 2)} секунду")
        else:
            uploaded_text = st.text(f"Видео будет обработано примерно за : {round(self.duration*3, 2)} секунд")
         
       # time.sleep(int(self.duration*5/4))
       # uploaded_text.empty()
       # uploaded_text = st.text(f"Осталось совсем чуть-чуть!")
       # time.sleep(int(self.duration*5/4))
       # uploaded_text.empty()
       # uploaded_text = st.text(f"Осталось примерно {int(self.duration-2*(self.duration*5/4))} секунд!")
       # time.sleep(int(self.duration*5/4))
       # uploaded_text.empty()
       # uploaded_text = st.text(f"Готово! Буквально через {int(self.duration)} секунд сможете скачать файл!")

start_time = time.time()
GREEN = (0, 255, 0)
model = YOLO("yolov8.onnx")

path_to_processed_video = "output_video.mp4"

center_x_list = []
center_y_list = []
frames = []
processing_done = False  # Flag to indicate if processing has been done

def create_video_writer(path, output_filename) -> None:
    """
    Функция принимает путь к видео и выдаёт объект для создания видео из кадров

    path: путь к видео
    output_filename: путь к месту, в котором мы хотим сохранить видео
    
    """
     
    video = mp.VideoFileClip(path)
    first_frame = video.get_frame(0)
    frame_width, frame_height = PIL.Image.fromarray(first_frame).size
    fps = video.fps

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    writer = cv2.VideoWriter(output_filename, fourcc, fps,
                             (frame_width, frame_height))

    return writer

def process_video(path):
    global center_x_list, center_y_list, frames, processing_done

    writer = create_video_writer(path, path_to_processed_video)
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
                predict = model(frame, task="detect")[0]

                if int(sv.Detections.from_ultralytics(predict).xyxy.size) == int(0):
                    print("Не найдено зрачка на фото")
                    writer.write(frame)
                else:
                    coordinates = copy.deepcopy(sv.Detections.from_ultralytics(predict).xyxy)
                    x1, y1, x2, y2 = int(coordinates[0][0]), int(coordinates[0][1]), int(coordinates[0][2]), int(coordinates[0][3])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 2)
                    image_np = cv2.cvtColor(np.array(PIL.Image.fromarray(predict.orig_img)), cv2.COLOR_RGB2BGR)

                    # считаем центр зрачка
                    center_x = ((coordinates[0][2]+coordinates[0][0]) / 2)
                    center_y = ((coordinates[0][3]+coordinates[0][1]) / 2)

                    center_x_list.append(center_x)
                    center_y_list.append(center_y)
                    frames.append(frame_number)

                    # отрисовываем прямоугольник
                    cv2.rectangle(image_np, (x1, y1), (x2, y2), GREEN, 2)

                    # отрисовываем центр зрачка
                    cv2.circle(image_np, (int(center_x), int(center_y)), radius=3, color=GREEN, thickness=-1)
                    writer.write(image_np)


                key = cv2.waitKey(20)

                if key == ord('q'):
                    print("Стоп")
                    break
                else:
                    print("Нажата кнопка:", key)
            else:
                print("Взятие видео закончено")
                break

        end_time = time.time()
        execution_time = end_time - start_time
        print("Обработка заняла:", execution_time, "секунд")

        video_cap.release()
        cv2.destroyAllWindows()

        # Save CSV file
        df = pd.DataFrame({'center_x': center_x_list, 'center_y': center_y_list, "frame": frames})
        csv_filename = path_to_processed_video.replace(".mp4", "_frames.csv")
        df.to_csv(csv_filename, index=False)

        processing_done = True

    if __name__ == "__main__":
        main()


def get_video_duration(path):
    video = mp.VideoFileClip(path)
    return video.duration
#st.text(f"Подождите примерно: {3*}")
        
csv_filename = path_to_processed_video.replace(".mp4", "_frames.csv")
uploaded_file_video = st.file_uploader("Загрузите сюда видео!", type="mp4")
 
if uploaded_file_video and not processing_done:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file_video.read())
        path = temp_file.name
        duration = get_video_duration(path)
        
    text = TextShower(duration=duration)
    text.show_text()

    process_video(path)
    st.write("Обработанное видео сохранено как 'output.mp4' ")
    st.download_button("Скачать обработанное видео", data=open(path_to_processed_video, "rb").read(), file_name=path_to_processed_video)
    st.download_button("Скачать файл с координатами", data=open(csv_filename, "rb").read(),  file_name=csv_filename)
elif processing_done:
    st.write("Обработка закончена. Если хотите обработать ещё одно видео, перезагрузите страницу")

