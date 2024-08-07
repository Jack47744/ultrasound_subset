import cv2
from SononetInference import SononetInference

def extract_planes_from_video(path, plane_detector):
    print("Extracting planes...")

    video = cv2.VideoCapture(path)
    success, frame = video.read()

    while success:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        confidence_score, pred_plane, saliency_map = plane_detector.detect_scan_planes(gray_frame)

        # use confidence_score, pred_plane or saliency_map for whatever you need to do.
        print(confidence_score, pred_plane)

        success, frame = video.read()

if __name__ == "__main__":
    path = ""
    sononet = SononetInference()
    extract_planes_from_video(path, sononet)