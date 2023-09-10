import cv2
from pixellib.semantic import semantic_segmentation
import pixellib
from pixellib.torchbackend.instance import instanceSegmentation

ins = instanceSegmentation()
# ins.load_model("pointrend_resnet50.pkl")
# ins.segmentImage("sample2.jpg", show_bboxes=True,
#                  output_image_name="output_image.jpg")


capture = cv2.VideoCapture(0)

ins.load_model("pointrend_resnet50.pkl",
               confidence=0.7, detection_speed="rapid")
ins.process_camera(capture, show_bboxes=True, overlay=True,
                   check_fps=True, show_frames=True, frame_name="FRAME", output_video_name="output.mp4")
