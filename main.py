import cv2
import numpy as np
from lxml import etree
def process_image(image_path):
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    section_height = height // 4 
    states = []
    for i in range(4):
        section = image[i * section_height:(i + 1) * section_height, :]
        hsv = cv2.cvtColor(section, cv2.COLOR_BGR2HSV)
        color = detect_color(hsv)
        face_type = detect_shape_type(section)
        state = determine_state(color, face_type)
        states.append(state)
    return states
# Função para detectar a cor predominante na seção
def detect_color(hsv_section):
    color_ranges = {
        'br': ([0, 0, 200], [180, 30, 255]),  
        'vr': ([35, 100, 100], [85, 255, 255]),  
        'am': ([20, 100, 100], [30, 255, 255]), 
        'vm': ([0, 100, 100], [10, 255, 255]),  
        'vzo': ([0, 0, 50], [180, 255, 100]), 
    }
    for color, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv_section, np.array(lower), np.array(upper))
        if cv2.countNonZero(mask) > 1000:  
            return color
    return 'unknow' 
def detect_shape_type(section):
    gray = cv2.cvtColor(section, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        _, _, angle = cv2.fitEllipse(contour) if len(contour) >= 5 else (0, 0, 0)
        if angle > 80 and angle < 100:
            return 's' 
        elif angle < 10 or angle > 170:
            return 'i'  
    return 'm' 
def determine_state(color, face_type):
    return color + face_type
def generate_xml(states, output_path="estado_jogo.xml"):
    root = etree.Element("EloMaluco")
    estado_atual = etree.SubElement(root, "EstadoAtual")
    for row_states in states:
        row_element = etree.SubElement(estado_atual, "row")
        for col_state in row_states:
            col_element = etree.SubElement(row_element, "col")
            col_element.text = col_state
    tree = etree.ElementTree(root)
    tree.write(output_path, pretty_print=True, xml_declaration=True, encoding="UTF-8")
    print(f"Arquivo XML gerado em: {output_path}")
def main():
    input_images = ["image1.png", "image2.png", "image3.png", "image4.png"]
    all_states = []
    for image_path in input_images:
        states = process_image(image_path)
        all_states.append(states)
    generate_xml(all_states)
if __name__ == "__main__":
    main()