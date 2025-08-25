import cv2
from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, frame):
        result = self.model(frame)[0]
        key_boxes = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = result.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            key_boxes.append({
                "label": label,
                "x": x1,
                "y": y1,
                "width": x2 - x1,
                "height": y2 - y1,
                "center_x": (x1 + x2) / 2,
                "center_y": (y1 + y2) / 2
            })
        return key_boxes
    
    # 키박스 안에 글자 표시
    def draw_boxes(self, image, boxes_data):
        """
        인식된 키보드 박스들을 이미지에 그리고, 라벨을 박스 중앙에 표시합니다.
        """
        for box_data in boxes_data:
            label = box_data.get('label', '')
            x = int(box_data.get('x', 0))
            y = int(box_data.get('y', 0))
            w = int(box_data.get('width', 0))
            h = int(box_data.get('height', 0))

            # 사각형 그리기 (보라색, 두께 2)
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 2)
            
            # 1. 텍스트 크기와 폰트 스케일 설정
            font_scale = 0.5
            font_thickness = 2
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

            # 2. 텍스트를 박스의 중앙에 위치시키기 위한 좌표 계산
            text_x = x + (w - text_w) // 2
            text_y = y + (h + text_h) // 2

            # 3. 계산된 위치에 텍스트 그리기
            cv2.putText(image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 255), font_thickness)


    # 키박스 위에 글자 표시    
    # def draw_boxes(self, image, boxes_data):
    #     """
    #     인식된 키보드 박스들을 이미지에 그립니다.
    #     """
    #     for box_data in boxes_data:
    #         label = box_data.get('label', '')
    #         x = int(box_data.get('x', 0))
    #         y = int(box_data.get('y', 0))
    #         w = int(box_data.get('width', 0))
    #         h = int(box_data.get('height', 0))

    #         # 사각형 그리기 (보라색, 두께 2)
    #         cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 2)
            
    #         # 라벨 텍스트 그리기
    #         cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)