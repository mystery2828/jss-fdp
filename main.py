import cv2
from pixellib.semantic import semantic_segmentation
import pixellib
from pixellib.torchbackend.instance import instanceSegmentation

ins = instanceSegmentation()
ins.load_model("pointrend_resnet50.pkl")
ins.segmentImage("sample2.jpg", show_bboxes=True,
                 output_image_name="output_image.jpg")
