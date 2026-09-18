import argparse
import os
import sys

import cv2
import numpy as np
from ultralytics import YOLO


VEHICLE_LABELS = {"car", "truck", "bus"}
PERSON_LABEL = "person"
MOTORCYCLE_LABEL = "motorcycle"

MAX_VEHICLE_COUNT = 60
MAX_PERSON_COUNT = 5
MAX_MOTORCYCLE_COUNT = 3
OVERCROWD_VEHICLE_COUNT = 8
OVERCROWD_PERSON_COUNT = 5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Smart Parking AI: YOLO-based parking lot video annotation with rule-based alerts."
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to the input video file."
    )
    parser.add_argument(
        "--output-dir",
        default="yolo_output",
        help="Directory where the processed video will be saved."
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLO model path or model name."
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.2,
        help="YOLO confidence threshold."
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="YOLO inference image size."
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display the processed video while running."
    )
    return parser.parse_args()


def draw_detections(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = map(int, det["box"])
        label = det["label"]
        conf = det["conf"]

        if label in VEHICLE_LABELS:
            color = (0, 255, 0)
        elif label == PERSON_LABEL:
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        text_y = max(y1 - 5, 15)
        cv2.putText(
            frame,
            f"{label} {conf:.2f}",
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )


def draw_alert_banner(frame, alerts):
    if not alerts:
        return

    requested_banner_height = 50 + len(alerts) * 30
    banner_height = min(requested_banner_height, frame.shape[0])

    overlay = np.zeros((banner_height, frame.shape[1], 3), dtype=np.uint8)
    overlay[:] = (0, 0, 255)

    alpha = 0.7
    frame[0:banner_height, 0:frame.shape[1]] = cv2.addWeighted(
        frame[0:banner_height, 0:frame.shape[1]],
        1 - alpha,
        overlay,
        alpha,
        0
    )

    max_lines = max((banner_height - 20) // 30, 0)
    for i, alert in enumerate(alerts[:max_lines]):
        cv2.putText(
            frame,
            alert,
            (20, 30 + i * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


def apply_rules(detections, frame):
    vehicle_count = sum(1 for det in detections if det["label"] in VEHICLE_LABELS)
    person_count = sum(1 for det in detections if det["label"] == PERSON_LABEL)
    motorcycle_count = sum(1 for det in detections if det["label"] == MOTORCYCLE_LABEL)

    alerts = []

    if vehicle_count > MAX_VEHICLE_COUNT:
        alerts.append("Too many vehicles in the parking lot! Consider restrictions.")

    if person_count > MAX_PERSON_COUNT:
        alerts.append("High pedestrian traffic detected. Caution advised.")

    if motorcycle_count > MAX_MOTORCYCLE_COUNT:
        alerts.append("High motorcycle count detected. Check parking compliance.")

    if vehicle_count > OVERCROWD_VEHICLE_COUNT and person_count > OVERCROWD_PERSON_COUNT:
        alerts.append("Critical Alert: possible overcrowding detected.")

    draw_alert_banner(frame, alerts)


def extract_detections(results, model):
    detections = []

    for result in results:
        for box in result.boxes:
            coords = box.xyxy.cpu().numpy()[0].tolist()
            cls_id = int(box.cls.cpu().numpy()[0])
            label = model.names[cls_id]
            conf = float(box.conf.cpu().numpy()[0])

            detections.append({
                "box": coords,
                "label": label,
                "conf": conf
            })

    return detections


def main():
    args = parse_args()

    if not os.path.isfile(args.video):
        print(f"Error: input video not found: {args.video}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "processed_parking_lot_video.mp4")

    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: could not open video: {args.video}")
        sys.exit(1)

    out = None

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if width <= 0 or height <= 0:
            print("Error: invalid video dimensions.")
            sys.exit(1)

        if fps <= 0:
            print("Warning: video FPS not detected. Using fallback FPS = 30.")
            fps = 30.0

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            print(f"Error: could not create output video: {output_path}")
            sys.exit(1)

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)
            detections = extract_detections(results, model)

            apply_rules(detections, frame)
            draw_detections(frame, detections)

            out.write(frame)
            frame_count += 1

            if args.display:
                cv2.imshow("Smart Parking AI", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        print(f"Processed {frame_count} frames.")
        print(f"Processed video saved at: {output_path}")

    finally:
        cap.release()

        if out is not None:
            out.release()

        if args.display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
