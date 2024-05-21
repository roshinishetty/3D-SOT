import cv2
import os

images_location = 'images'
vid_file = 'video_0019-5.mp4'

images = []
for im in os.listdir(images_location):
    if im.endswith(".png") or im.endswith(".jpg"):
        images.append(im)

frame = cv2.imread(os.path.join(images_location, images[0]))
height, width, _ = frame.shape
print(height,width)
#custom variables to manipulate image dimensions for stacked video creation
width=1280
# height=674
video_writer = cv2.video_writerWriter(vid_file, 0, 10, (width,height))

def sortbybatch(e):
    return int(e.split('.')[0])

images.sort(key=sortbybatch)

for image in images:
    print(image)
    im = cv2.imread(os.path.join(images_location, image))
    im = cv2.resize(im, (width, height))
    video_writer.write(im)

cv2.destroyAllWindows()
video_writer.release()