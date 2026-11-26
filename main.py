"""
Automatic License Plate Recognition (ALPR) & Tracking Pipeline
Combines YOLOv8 for vehicle plate localization, EasyOCR for text extraction,
regex-based character error-correction, and temporal smoothing across video frames.
"""

import argparse
import re
from collections import defaultdict, deque
import cv2
import numpy as np
import easyocr
from ultralytics import YOLO


def parse_arguments():
    parser = argparse.ArgumentParser(description="Real-Time Automatic License Plate Recognition (ALPR)")
    parser.add_argument("--input", type=str, default="0", help="Path to input video file or webcam index (default: 0)")
    parser.add_argument("--output", type=str, default=None, help="Path to save processed output video (e.g. output.mp4)")
    parser.add_argument("--model", type=str, default="license_plate_best.pt", help="Path to trained YOLO license plate model")
    parser.add_argument("--conf", type=float, default=0.3, help="YOLO confidence detection threshold")
    parser.add_argument("--gpu", action="store_true", default=True, help="Use GPU acceleration for OCR")
    return parser.parse_args()


# Heuristic character mapping for typical OCR confusion
MAPPING_NUM_TO_ALPHA = {"0": "O", "1": "I", "5": "S", "8": "B"}
MAPPING_ALPHA_TO_NUM = {"O": "0", "I": "1", "Z": "2", "S": "5", "B": "8", "R": "5"}
PLATE_PATTERN = re.compile(r"^[A-Z]{2}[0-9]{2}[A-Z]{3}$")


def correct_number_plate(ocr_text: str) -> str:
    """
    Applies domain-specific alphanumeric disambiguation based on known plate format:
    Positions 0-1: Alphabetic
    Positions 2-3: Numeric
    Positions 4-6: Alphabetic
    """
    ocr_text = ocr_text.upper().replace(" ", "").replace("-", "")
    if len(ocr_text) != 7:
        return ocr_text

    corrected = []
    for i, ch in enumerate(ocr_text):
        if i < 2 or i >= 4:
            # Expected Alphabetic
            if ch.isdigit() and ch in MAPPING_NUM_TO_ALPHA:
                corrected.append(MAPPING_NUM_TO_ALPHA[ch])
            elif ch.isalpha():
                corrected.append(ch)
            else:
                return ""
        else:
            # Expected Numeric
            if ch.isalpha() and ch in MAPPING_ALPHA_TO_NUM:
                corrected.append(MAPPING_ALPHA_TO_NUM[ch])
            elif ch.isdigit():
                corrected.append(ch)
            else:
                return ""
    return "".join(corrected)


def recognize_plate(plate_crop: np.ndarray, reader: easyocr.Reader) -> str:
    """Preprocesses cropped plate region and extracts text via EasyOCR."""
    if plate_crop is None or plate_crop.size == 0:
        return ""

    # Preprocessing: Grayscale -> Otsu Thresholding -> Bicubic Upsampling
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    plate_resized = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    try:
        ocr_result = reader.readtext(
            plate_resized,
            detail=0,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )
        if ocr_result:
            candidate = "".join(ocr_result)
            corrected = correct_number_plate(candidate)
            return corrected if corrected else candidate
    except Exception:
        pass
    return ""


def get_box_id(x1: int, y1: int, x2: int, y2: int) -> str:
    """Generates a spatial bucket pseudo-ID for tracking across frames."""
    return f"{int(x1 / 15)}_{int(y1 / 15)}_{int(x2 / 15)}_{int(y2 / 15)}"


def get_stable_plate(box_id: str, new_text: str, plate_history: dict, plate_final: dict) -> str:
    """Applies majority voting over recent history buffer for temporal stability."""
    if new_text:
        plate_history[box_id].append(new_text)
        most_common = max(set(plate_history[box_id]), key=plate_history[box_id].count)
        plate_final[box_id] = most_common
    return plate_final.get(box_id, "")


def main():
    args = parse_arguments()

    print("[INFO] Loading YOLO model...")
    model = YOLO(args.model)

    print("[INFO] Initializing EasyOCR...")
    reader = easyocr.Reader(['en'], gpu=args.gpu)

    # Initialize video capture (numeric index if webcam, string if file path)
    video_source = int(args.input) if args.input.isdigit() else args.input
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {args.input}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    out = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"[INFO] Writing output video to: {args.output}")

    plate_history = defaultdict(lambda: deque(maxlen=10))
    plate_final = {}

    print("[INFO] Starting real-time processing. Press 'q' to exit.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=args.conf, verbose=False)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])

                # Ensure crop coordinates remain within image boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)

                plate_crop = frame[y1:y2, x1:x2]
                raw_text = recognize_plate(plate_crop, reader)

                box_id = get_box_id(x1, y1, x2, y2)
                stable_text = get_stable_plate(box_id, raw_text, plate_history, plate_final)

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw plate label banner
                display_label = stable_text if stable_text else f"Plate ({conf:.2f})"
                (text_w, text_h), _ = cv2.getTextSize(display_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (x1, y1 - 25), (x1 + text_w, y1), (0, 255, 0), -1)
                cv2.putText(frame, display_label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        if out:
            out.write(frame)

        cv2.imshow("Automatic License Plate Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    print("[INFO] Processing finished.")


if __name__ == "__main__":
    main()
