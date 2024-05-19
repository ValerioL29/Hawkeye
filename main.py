import cv2
import torch

from hawkeye.holoh import HOLOH
from hawkeye.utils import AGENT_CFG_DIR, MODELS_DIR, DATA_DIR

# Initialize the agent
device = torch.device("mps")
holoh_agent = HOLOH(config=AGENT_CFG_DIR, device=device)\
    .load(model_weights_paths=MODELS_DIR / "holoh")

# Perform tracking with the agent
holoh_agent.run_aio(video_or_stream=DATA_DIR / "test.mp4")

# # Build a YOLOv8l model from pretrained weight
# model = YOLO(MODELS_DIR / "cfg" / "yolov8" / "yolov8l.yaml")\
#     .load(MODELS_DIR / "yolov8" / "yolov8l.pt")
# seg_model = YOLO(MODELS_DIR / "cfg" / "yolov8" / "yolov8l-seg.yaml")\
#     .load(MODELS_DIR / "yolov8" / "yolov8l-seg.pt")
# byte_tracker_cfg = MODELS_DIR / "cfg" / "trackers" / "bytetrack.yaml"
#
#
# # Perform detection with the model
# # Open the video file
# video_path = DATA_DIR / "test.mp4"
# cap = cv2.VideoCapture(str(video_path))
#
# # Loop through the video frames
# while cap.isOpened():
#     # Read a frame from the video
#     success, frame = cap.read()
#
#     if success:
#         # Run YOLOv8 inference on the frame
#         # results = model.track(frame, conf=0.3, iou=0.3, persist=True, tracker=byte_tracker_cfg)
#         seg_results = seg_model.track(frame, conf=0.3, iou=0.3, persist=True, tracker=byte_tracker_cfg)
#
#         # Visualize the results on the frame
#         # annotated_frame = results[0].plot(conf=False, labels=False)
#         annotated_frame = seg_results[0].plot(conf=False, labels=False, boxes=False)
#
#         # Display the annotated frame
#         cv2.imshow("YOLOv8 Tracking", annotated_frame)
#
#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#     else:
#         # Break the loop if the end of the video is reached
#         break
#
# # Release the video capture object and close the display window
# cap.release()
# cv2.destroyAllWindows()
