from moviepy.editor import VideoFileClip
import moviepy.editor as mp
import PIL
from time import time
import cv2

#декоратор, который позволяет сколько времени функция выполняла свою задачу
def timer(func):
    def wrapper(*args, **kwargs):
        t1 = time()
        result = func(*args, *kwargs)
        t2 = time() 
        print(f"{func.__name__} executed in {(t2-t1):.4f}")
        return result
    return wrapper

def cut_video(path: str, start: int, end: int, savepath: str) -> None:
    """
    Функция, позволяющая вырезать часть видео и сохранить его

    path: путь к видео, которое Вы хотите обрезать
    start: секунда с которой будет обрезано видео 
    end: секунда до которой будет обрезано видео
    savepath: место, в которое сохраняем обработанное видео

    """

    clip = VideoFileClip(path)
    clip_video = clip.subclip((start),(end))
    clip_video.write_videofile(savepath, codec="libx264")
    return clip_video

@timer
def get_fps(path) -> int:
    """
    Функция принимает путь к видео и выдаёт количество кадров в секунду(fps) в целых числах

    path: путь к видео
    
    """
    # png, jpeg if another smth
    video_cap = cv2.VideoCapture(path)
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))
    return int(fps)

@timer
def get_first_frame_info(path) -> None:
    """
    Функция принимает путь к видео и выдаёт размер первого кадра

    path: str
    
    """

    video = mp.VideoFileClip(path)
    first_frame = video.get_frame(0)
    image = PIL.Image.fromarray(first_frame) 
    return image.size


def cut_video_on_frames(path:str, path_to_save: str) -> None:
    """
    Функция принимает путь к видео и каждый второй кадр добавляет в директорию, которая была указана в аргументах

    path: путь к видео, которое Вы хотите обрезать
    path_to_save: место, в которое сохраняем обработанное видео
    """
    capture = cv2.VideoCapture(path)
    frame_number = 0
    frame_2 = 0
    while(True):

        success, frame = capture.read()

        if success:
            if frame_number%2 == 0:
                frame_2 +=1
                cv2.imwrite(f'{frame_2}.jpg', frame)

        else:
            break

        frame_number = frame_number+1

    capture.release()

    frame_number = 1
    print(f'{path_to_save}/frame_{frame_number}.jpg')

 
