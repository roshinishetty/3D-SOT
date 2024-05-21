# from moviepy.editor import VideoFileClip

# videoClip = VideoFileClip("video_im0000.mp4")

# videoClip.write_gif("video_im0000.gif")

import cv2
import glob
import os
import shutil
from PIL import Image

images = glob.glob(f"images_vid1/*.png")
print(images)
def sortbybatch(e):
    return int(e.split('_')[-1].split('.')[0])

images.sort(key=sortbybatch)

print(images)

frames =[]
for im in images:
    frames.append(Image.open(im))
first_frame = frames[0]
first_frame.save("./vid1_.gif", format="GIF", append_images=frames,
                   save_all=True, duration=50, loop=0)