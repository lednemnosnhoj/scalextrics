import time
import picamera
import numpy as np
import cv2
from collections import deque

# Configuration
RECORD_TIME_BEFORE = 3  # Time to record before motion detection in seconds
RECORD_TIME_AFTER = 3   # Time to record after motion detection in seconds
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAMERATE = 24
TOTAL_RECORD_TIME = 60  # Total record time in seconds

# Need to use a circular buffer: https://picamera.readthedocs.io/en/release-1.13/recipes1.html#recording-to-a-circular-stream

def detect_motion(frame1, frame2):
    # Convert frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    # Compute absolute difference between the frames
    diff = cv2.absdiff(gray1, gray2)
    # Threshold the difference to get binary image
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    # Find contours to determine if there's significant motion
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours) > 0

def main():
    with picamera.PiCamera(resolution=(FRAME_WIDTH, FRAME_HEIGHT), framerate=FRAMERATE) as camera:
        camera.start_preview()
        time.sleep(2)  # Warm-up time for the camera
        
        # Prepare buffers for motion detection and video recording
        prev_frame = np.empty((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        current_frame = np.empty((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        camera.capture(prev_frame, 'bgr')
        
        frame_buffer = deque(maxlen=RECORD_TIME_BEFORE * FRAMERATE)
        recording = False
        motion_detected_time = None

        # Initialize VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, FRAMERATE, (FRAME_WIDTH, FRAME_HEIGHT))

        start_time = time.time()

        while time.time() - start_time < TOTAL_RECORD_TIME:
            # Capture current frame
            camera.capture(current_frame, 'bgr')
            frame_buffer.append(current_frame.copy())

            if detect_motion(prev_frame, current_frame):
                if not recording:
                    recording = True
                    motion_detected_time = time.time()
                    print("Motion detected, starting recording")

                # Save video frames before and after motion detection
                while time.time() - motion_detected_time <= RECORD_TIME_AFTER:
                    if not frame_buffer:
                        break
                    frame = frame_buffer.popleft()
                    out.write(frame)  # Save frame to video file
                    print("Saving frame...")

                recording = False
                print("Recording stopped")

            # Prepare for next frame
            prev_frame = np.copy(current_frame)
            time.sleep(1.0 / FRAMERATE)  # Adjust sleep time to match frame rate

        # Release the VideoWriter
        out.release()
        print("Recording completed and saved to output.avi")

if __name__ == "__main__":
    main()
