import numpy as np
import cv2
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

#DETECTOR DE CORES
def detect_colors(path, imageResult):
    imageFrame = cv2.imread(path, cv2.IMREAD_COLOR)
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

    colors = {
        "vm": [
            (np.array([0, 50, 0], np.uint8), np.array([10, 255, 255], np.uint8)),
            (np.array([160, 50, 0], np.uint8), np.array([180, 255, 255], np.uint8))
        ],
        "vr": [(np.array([31, 52, 72], np.uint8), np.array([90, 255, 255], np.uint8))],
        "am": [(np.array([20, 100, 100], np.uint8), np.array([30, 255, 255], np.uint8))],
        "br": [(np.array([0, 0, 200], np.uint8), np.array([180, 20, 255], np.uint8))]
    }

    detected_colors = []

    for color, ranges in colors.items():
        mask = None
        for lower, upper in ranges:
            if mask is None:
                mask = cv2.inRange(hsvFrame, lower, upper)
            else:
                mask = cv2.bitwise_or(mask, cv2.inRange(hsvFrame, lower, upper))

        mask = cv2.dilate(mask, np.ones((5, 5), "uint8"))
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:
                _, y, _, _ = cv2.boundingRect(contour)
                detected_colors.append((color, y))

    detected_colors.sort(key=lambda x: x[1])
    color_array = [color for color, _ in detected_colors[:4]]
    if(len(color_array)< 4):
        voidIndex = imageResult.index("vzo")
        color_array.insert(voidIndex, "vzo")
    result = [prefixo if prefixo == sufixo else prefixo + sufixo for prefixo, sufixo in zip(color_array, imageResult)]
    return result
    
#DETECTOR DE PADRÂO
def padronizar_imagem(imagem, target_size=200):
    h, w = imagem.shape[:2]
    gray_value = 128
    mask = imagem > 0 
    uniform_img = np.zeros_like(imagem)
    uniform_img[mask] = gray_value
    scale = min(target_size / w, target_size / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(uniform_img, (new_w, new_h))
    square_image = np.zeros((target_size, target_size), dtype=np.uint8)
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    square_image[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return square_image

def calculate_similarity(image1, image2):
    image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))
    
    difference = cv2.absdiff(image1, image2)
    similarity = np.mean(difference)
    return similarity


def load_pattern_images():
    upper = cv2.imread('./images/UperElo.png', cv2.IMREAD_GRAYSCALE)
    middle = cv2.imread('./images/MiddleElo.png', cv2.IMREAD_GRAYSCALE)
    lower = cv2.imread('./images/LowerElo.png', cv2.IMREAD_GRAYSCALE)
    void = cv2.imread('./images/void.png', cv2.IMREAD_GRAYSCALE)
    if upper is None or middle is None or lower is None:
        print("Erro ao carregar as imagens-alvo.")
        exit(1)
    upper = padronizar_imagem(upper, 74)
    middle = padronizar_imagem(middle, 74)
    lower = padronizar_imagem(lower, 74)
    void = padronizar_imagem(void, 74)
    return [upper, middle, lower, void]

def detect_pattern(path):
    [upper, middle, lower, void] = load_pattern_images()
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    binary = cv2.adaptiveThreshold(
        image, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        15, 3
    )

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    widths = [cv2.boundingRect(contour)[2] for contour in contours]
    mean_width = np.mean(widths) if widths else 0

    min_width = int(mean_width * 0.8)
    min_area = 500

    target_size = 200

    processed_elements = []

    for idx, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        
        area = cv2.contourArea(contour)
        if cv2.contourArea(contour) > 500:
            mask = np.zeros_like(image)
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

            cropped = cv2.bitwise_and(image[y:y+h, x:x+w], image[y:y+h, x:x+w], mask=mask[y:y+h, x:x+w])

            h_cropped, w_cropped = cropped.shape[:2]
            mask_flood = np.zeros((h_cropped + 2, w_cropped + 2), np.uint8)
            cv2.floodFill(cropped, mask_flood, (0, 0), 0)

        if w >= min_width and area >= min_area:
            
            gray_value = 128
            mask = cropped > 0 

            uniform_img = np.zeros_like(cropped)
            uniform_img[mask] = gray_value

            scale = min(target_size / w, target_size / h) 
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            resized = cv2.resize(uniform_img, (new_w, new_h))
            processed_elements.append(resized)

    templates = [upper, middle, lower, void]
    labels = ["s", "m", "i", "vzo"]
    patternResult = []

    for _, elem in enumerate(processed_elements):
        h, w = elem.shape
        segment_height = h // 4
        for j in range(4):
            y_start = j * segment_height
            y_end = (j + 1) * segment_height if j < 3 else h
            segment = elem[y_start:y_end, :]
            melhor_label = None
            menor_diferenca = float('inf')
            for idx, template in enumerate(templates):
                similarity = calculate_similarity(template, segment)
                if similarity < menor_diferenca:
                    menor_diferenca = similarity
                    melhor_label = labels[idx]
            
            patternResult.append((melhor_label))

    return patternResult

def matriz_para_xml(matriz):
    # Transposição da matriz, pois as faces se representam em diferentes colunas
    matriz_transposta = list(map(list, zip(*matriz)))
    
    elo_maluco = ET.Element("EloMaluco")
    estado_atual = ET.SubElement(elo_maluco, "EstadoAtual")
    
    for linha in matriz_transposta:
        row = ET.SubElement(estado_atual, "row")
        for item in linha:
            col = ET.SubElement(row, "col")
            col.text = item
    
    xml_str = ET.tostring(elo_maluco, encoding='utf-8')
    xml_pretty = minidom.parseString(xml_str).toprettyxml(indent="  ", encoding='utf-8')
    return xml_pretty.decode('utf-8')

def main():
    EloResult = []
    image_input = input("Qual input de imagem você deseja fazer o xml?(1, 2 ou 3)")
    images = [('./images/Ex_input0' + image_input + '_0' + str(index) + '.png') for index in range(4, 0, -1)]

    for image in images:
        EloResult.append(detect_colors(image, detect_pattern(image)))

    xml_final = matriz_para_xml(EloResult)

    with open('output0' + image_input + '.xml', "w", encoding="utf-8") as file:
        file.write(xml_final)

    

if __name__ == "__main__":
    main()