import os
import cv2
from inference_sdk import InferenceHTTPClient
from inference_sdk.webrtc import WebcamSource, StreamConfig, VideoMetadata

# Initialize client using environment variable
client = InferenceHTTPClient.init(
    api_url="https://serverless.roboflow.com",
    api_key=os.getenv("ROBOFLOW_API_KEY")
)

# Webcam source
source = WebcamSource(resolution=(1280, 720))

# Streaming configuration
config = StreamConfig(
    stream_output=["output_image"],
    data_output=["count_objects", "predictions"],
    requested_plan="webrtc-gpu-medium",
    requested_region="us",
    processing_timeout=600,
)

# Start WebRTC stream
session = client.webrtc.stream(
    source=source,
    workflow="detect-count-and-visualize-7",
    workspace="asep81",
    image_input="image",
    config=config
)

@session.on_frame
def on_frame(frame, metadata):
    cv2.imshow("Roboflow Inference Output", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        session.close()

@session.on_data
def on_data(data: dict, metadata: VideoMetadata):
    print(f"Frame {metadata.frame_id}: {data}")

session.run()
