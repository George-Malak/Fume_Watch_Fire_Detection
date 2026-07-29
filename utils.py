import cv2
import numpy as np
import subprocess
import os

class FumeWatchProcessor:
    def __init__(self, target_size=(640, 640)):
        self.target_size = target_size

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        resize -> reduce saturation via HSV -> return processed image
        """
        # 1. Resize
        resized_img = cv2.resize(image, self.target_size)

        # 2. Reduce saturation via HSV
        hsv = cv2.cvtColor(resized_img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.6, 0, 255)
        processed_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return processed_img

    def predict_image(self, image: np.ndarray, model, conf_threshold=0.25):
        """
        Preprocess -> Predict -> Plot Bounding Boxes
        """
        # 1. process
        processed_img = self.preprocess_image(image)

        # 2. predict
        results = model.predict(source=processed_img, conf=conf_threshold, verbose=False)

        # 3. use results and show box
        boxes = results[0].boxes
        annotated_img = self.show_box(processed_img, boxes, is_model=True)

        return annotated_img, results[0]

    def process_video(self, video_path, output_path, model, conf_threshold=0.25):
        """
        Read Frame -> Preprocess -> Predict -> Write --> Save Video 
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video stream or file: {video_path}")
            return False

        # Get video properties
        out_w, out_h = self.target_size
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0:
            fps = 30

        temp_raw = "temp_raw_video.mp4"
        # resize to target size
        fourcc = cv2.VideoWriter_fourcc(*"mp4")
        out = cv2.VideoWriter(temp_raw, fourcc, fps, (out_w, out_h), isColor=True)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 1. Preprocess
            processed_frame = self.preprocess_image(frame)

            # 2. Predict
            results = model.predict(source=processed_frame, conf=conf_threshold, verbose=False)

            # 3. Plot Bounding Boxes 
            boxes = results[0].boxes
            annotated_frame = self.show_box(processed_frame, boxes, is_model=True)

            # 4. Write frame
            out_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            out.write(out_frame)

        cap.release()
        out.release()
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    temp_raw,
                    "-vcodec",
                    "libx264",
                    output_path,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            if os.path.exists(temp_raw):
                os.remove(temp_raw)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error during ffmpeg processing: {e}")
            return False

    def show_box(self, image_input, label_data, is_model=False):
        """
        draw bounding boxes on img or video frame
        """
        if isinstance(image_input, str):
            img = cv2.imread(image_input)
            if img is None:
                return None
        else:
            img = image_input.copy()

        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w, _ = img.shape
        lines = []

        if not is_model:
            if isinstance(label_data, str) and os.path.exists(label_data):
                with open(label_data, "r") as f:
                    lines = [line.split() for line in f.readlines()]
        else:
            boxes = label_data
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                lines.append([cls, x1, y1, x2, y2, conf, True])

        for line in lines:
            if len(line) == 7:
                cls, x1, y1, x2, y2, conf, _ = line
            else:
                cls = int(line[0])
                x_c, y_c, box_w, box_h = map(float, line[1:])
                x1 = int((x_c - box_w / 2) * w)
                y1 = int((y_c - box_h / 2) * h)
                x2 = int((x_c + box_w / 2) * w)
                y2 = int((y_c + box_h / 2) * h)
                conf = None

            color = (255, 0, 0) if cls == 1 else (0, 0, 255) # Red for Fire, Blue for Smoke
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            label_name = "Smoke" if cls == 0 else "Fire"
            text = f"{label_name} {conf:.2f}" if conf is not None else label_name

            y1_text = max(int(y1) - 10, 15)
            cv2.putText(img, text, (int(x1), y1_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return img