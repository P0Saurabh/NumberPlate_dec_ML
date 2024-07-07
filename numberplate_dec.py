import cv2
import numpy as np
import tensorflow as tf
import pytesseract

model_path = "...\model_path"

model = tf.saved_model.load(model_path)

interpreter = tf.lite.Interpreter(model_path="ssd_mobilenet_v2_coco.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

label_map = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 6: 'bus', 7: 'train', 8: 'truck'
}


def detect_vehicle(image):
    input_shape = input_details[0]['shape']
    input_data = np.expand_dims(cv2.resize(image, (input_shape[1], input_shape[2])), axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.float32))

    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]
    num_detections = int(interpreter.get_tensor(output_details[3]['index'])[0])

    for i in range(num_detections):
        if scores[i] > 0.5:
            class_id = int(classes[i])
            class_name = label_map.get(class_id, 'N/A')
            if class_name in ['car', 'motorcycle']:
                ymin, xmin, ymax, xmax = boxes[i]
                xmin = int(xmin * image.shape[1])
                xmax = int(xmax * image.shape[1])
                ymin = int(ymin * image.shape[0])
                ymax = int(ymax * image.shape[0])
                return class_name, (xmin, ymin, xmax, ymax)
    return None, None


def detect_color(image, box):
    xmin, ymin, xmax, ymax = box
    vehicle_image = image[ymin:ymax, xmin:xmax]
    avg_color_per_row = np.average(vehicle_image, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    color = 'Unknown'

    if np.argmax(avg_color) == 0:
        color = 'Blue'
    elif np.argmax(avg_color) == 1:
        color = 'Green'
    elif np.argmax(avg_color) == 2:
        color = 'Red'

    return color


def detect_and_read_license_plate(image, box):
    xmin, ymin, xmax, ymax = box
    vehicle_image = image[ymin:ymax, xmin:xmax]

    gray = cv2.cvtColor(vehicle_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(blur, 30, 200)

    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = w / h

        if 2 < aspect_ratio < 5:  # Assuming license plates are rectangular
            plate_image = vehicle_image[y:y + h, x:x + w]
            text = pytesseract.image_to_string(plate_image, config='--psm 8')
            return text.strip()
    return None


def main(image_path):
    image = cv2.imread(image_path)

    vehicle_type, box = detect_vehicle(image)
    if vehicle_type:
        print(f"Detected a {vehicle_type}")

        color = detect_color(image, box)
        print(f"Color: {color}")

        plate_number = detect_and_read_license_plate(image, box)
        if plate_number:
            print(f"License Plate: {plate_number}")
        else:
            print("License Plate not detected.")
    else:
        print("No vehicle detected.")


if __name__ == "__main__":
    image_path = 'images.jpeg'  # Adjust to your image path
    main(image_path)