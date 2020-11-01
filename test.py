import cv2
import numpy as np
from PIL import Image

from smile import logging
from smile import app

from models.video_models import VideoGraphNet

def main(_):
    # image_path = "/mnt/data/m2cai/m2cai_tool/images/train/tool_video_01/tool_video_01_000025_0000000.jpg"
    # img = np.array(Image.open(image_path))
    # Image.fromarray(img).save("test/origin.jpg")
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Image.fromarray(gray).save("test/gray.jpg")
    # _, thresh = cv2.threshold(gray, 32, 255, cv2.THRESH_BINARY)
    # # Image.fromarray(thresh).save("test/thresh.jpg")
    # # logging.info()
    # idx = np.where(thresh == 255)
    # x_start = np.min(idx[0])
    # x_end = np.max(idx[0]) + 1
    # y_start = np.min(idx[1])
    # y_end = np.max(idx[1]) + 1
    # crop = img[x_start:x_end, y_start:y_end]
    # resized = cv2.resize(crop, (112, 112))
    # Image.fromarray(crop).save("test/crop.jpg")
    # Image.fromarray(resized).save("test/resized.jpg")
    

if __name__ == "__main__":
    app.run()
