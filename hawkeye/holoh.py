import cv2
import threading
from pathlib import Path
from typing import Union, Optional

import numpy as np
import torch
from ultralytics import YOLO

from hawkeye.utils import logger, MODELS_DIR


class HOLOH:
    """Hawkeye Helmet security agent powered by YOLO"""
    object_track_model: YOLO
    drivable_seg_model: YOLO
    lane_seg_model: YOLO
    loaded: bool = False

    def __init__(self, config: Union[dict, str, Path], device: torch.device = None):
        """Initialize the agent with configuration"""
        # Setup models
        self.setup_models(config)
        # Set device
        device_str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device or torch.device(device_str)
        emoji = "🚀" if self.device.type == "cuda" else "🔥" if self.device.type == "mps" else "🐢"
        logger.info(f"Using '{emoji + device_str}' for inferencing!")

    def setup_models(self, model_config_or_path: Union[dict, str, Path]):
        """Setup models for the agent"""
        if isinstance(model_config_or_path, (str, Path)):
            # Path check
            if not isinstance(model_config_or_path, Path):
                model_config_or_path = Path(model_config_or_path)
            if not model_config_or_path.exists():
                raise FileNotFoundError(f"Model configuration not found: {model_config_or_path}")

            # Initialize models
            self.object_track_model = YOLO(model_config_or_path / "holoh-detect.yaml")
            self.drivable_seg_model = YOLO(model_config_or_path / "holoh-drivable.yaml")
            self.lane_seg_model = YOLO(model_config_or_path / "holoh-lane.yaml")
        elif isinstance(model_config_or_path, dict):
            # Initialize models with path dictionary
            self.object_track_model = YOLO(model_config_or_path["detect"])
            self.drivable_seg_model = YOLO(model_config_or_path["drivable"])
            self.lane_seg_model = YOLO(model_config_or_path["lane"])
        else:
            raise ValueError("Invalid model configuration")

    def load(self, model_weights_paths: Path):
        """Load pretrained model weights"""
        self.object_track_model.load(model_weights_paths / "holoh-detect.pt").to(self.device, non_blocking=True)
        self.drivable_seg_model.load(model_weights_paths / "holoh-drivable.pt").to(self.device, non_blocking=True)
        self.lane_seg_model.load(model_weights_paths / "holoh-lane.pt").to(self.device, non_blocking=True)
        self.loaded = True

        return self

    @staticmethod
    def run_tracker_in_thread(filename: Union[str, Path], submodule: YOLO, task_name: str):
        """
        Runs a video file or webcam stream concurrently with the YOLOv8 model using threading.

        This function captures video frames from a given file or camera source and utilizes the YOLOv8 model for object
        tracking. The function runs in its own thread for concurrent processing.

        Args:
            filename (str): The path to the video file or the identifier for the webcam/external camera source.
                            Path to video file, 0 for webcam, 1 for external camera
            submodule (YOLO): The HOLOH model object.
            task_name (str): Thread for specific task

        Note:
            Press 'q' to quit the video display window.
        """
        # OpenCV can only accept string paths
        if isinstance(filename, Path):
            filename = str(filename)
        video = cv2.VideoCapture(filename)  # Read the video file

        while True:
            ret, frame = video.read()  # Read the video frames

            # Exit the loop if no more frames in either video
            if not ret:
                break

            # Track objects in frames if available
            results = submodule.track(frame, persist=True)
            res_plotted = results[0].plot()
            cv2.imshow(f"Tracking_Stream_{task_name}", res_plotted)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

        # Release video sources
        video.release()

    def run_multithread(self, video_or_stream: Union[str, Path, int]):
        """Predict with the agent"""
        # Check file path
        if isinstance(video_or_stream, str):
            video_or_stream = Path(video_or_stream)
        if isinstance(video_or_stream, Path) and not video_or_stream.exists():
            raise FileNotFoundError(f"Video not found: {video_or_stream}")

        # Check if models are loaded
        if not self.loaded:
            raise RuntimeError("Models are not loaded")

        # Create the tracker threads
        object_tracker_thread = threading.Thread(
            target=self.run_tracker_in_thread,
            args=(video_or_stream, self.object_track_model, "Blind Spot Detection"),
            daemon=True
        )
        drivable_seg_thread = threading.Thread(
            target=self.run_tracker_in_thread,
            args=(video_or_stream, self.drivable_seg_model, "Drivable Area Segmentation"),
            daemon=True
        )
        lane_seg_thread = threading.Thread(
            target=self.run_tracker_in_thread,
            args=(video_or_stream, self.lane_seg_model, "Lane Segmentation"),
            daemon=True
        )
        # Start the tracker threads
        object_tracker_thread.start()
        drivable_seg_thread.start()
        lane_seg_thread.start()

        # Wait for the tracker threads to finish
        object_tracker_thread.join()
        drivable_seg_thread.join()
        lane_seg_thread.join()

        # Clean up and close windows
        cv2.destroyAllWindows()

        return True

    def run_aio(
            self,
            video_or_stream: Union[str, Path, int],
            is_blind_spot: bool = False,
            window_name: str = "Hawkeye",
            output_path: Optional[Union[str, Path]] = None
    ):
        """Predict with the agent and save the video"""
        # Check file path
        if isinstance(video_or_stream, str):
            video_or_stream = Path(video_or_stream)
        if isinstance(video_or_stream, Path) and not video_or_stream.exists():
            raise FileNotFoundError(f"Video not found: {video_or_stream}")

        # Check if models are loaded
        if not self.loaded:
            raise RuntimeError("Models are not loaded")

        # OpenCV can only accept string paths
        video_source = str(video_or_stream)
        video = cv2.VideoCapture(video_source)  # Read the video file
        byte_tracker_cfg = MODELS_DIR / "cfg" / "trackers" / "bytetrack.yaml"

        out = None
        if output_path is not None:
            # Get video properties
            frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(video.get(cv2.CAP_PROP_FPS))

            # Initialize VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        # Time container
        inference_times = []
        while True:
            ret, frame = video.read()  # Read the video frames

            # Exit the loop if no more frames in either video
            if not ret:
                break

            if is_blind_spot:
                e1 = cv2.getTickCount()  # start
                # Only use the object tracker for blind spot detection
                detect_results = self.object_track_model.track(
                    frame, persist=True,
                    conf=0.3, iou=0.3,
                    tracker=byte_tracker_cfg
                )
                annotated_frame = detect_results[0].plot(conf=False, labels=False)
                e2 = cv2.getTickCount()  # end
            else:
                e1 = cv2.getTickCount()  # start
                # Track objects in frames if available
                detect_results = self.object_track_model.track(
                    frame, persist=True,
                    conf=0.3, iou=0.3,
                    tracker=byte_tracker_cfg
                )
                drivable_results = self.drivable_seg_model.track(
                    frame, persist=True,
                    conf=0.3, iou=0.3,
                    tracker=byte_tracker_cfg
                )
                lane_results = self.lane_seg_model.track(
                    frame, persist=True,
                    conf=0.3, iou=0.3,
                    tracker=byte_tracker_cfg
                )

                # Plot the results for visualization
                detect_plot = detect_results[0].plot(conf=False, labels=False)
                drivable_plot = drivable_results[0].plot(conf=False, img=detect_plot, boxes=False)
                annotated_frame = lane_results[0].plot(conf=False, img=drivable_plot, boxes=False)
                e2 = cv2.getTickCount()  # end

            # Write the frame into the video
            if output_path is not None:
                out.write(annotated_frame)

            # Display the results
            cv2.imshow(winname=window_name, mat=annotated_frame)
            inference_elapsed = (e2 - e1) / cv2.getTickFrequency()
            inference_times.append(inference_elapsed)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

        # Release video sources
        video.release()
        # Release the VideoWriter
        if output_path is not None:
            out.release()
        # Clean up and close windows
        cv2.destroyAllWindows()

        logger.info(f"Average inference time: {np.mean(inference_times):.4f} seconds")
        # For BOT-SORT Tracker - Average inference time: 0.0884 seconds per frame
        # For ByteTrack Tracker - Average inference time: 0.0479 seconds per frame

        return
