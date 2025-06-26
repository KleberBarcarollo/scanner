import cv2
from matplotlib import pyplot as plt

# Função para exibir imagens
def showImage(img, title="Imagem", size=(10, 8)):
    plt.figure(figsize=size)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

# Caminho da imagem
img_path = r"D:\img\11.jpg"

# Carrega imagem
img = cv2.imread(img_path)
assert img is not None, "A imagem não pôde ser lida. Verifique o caminho."

# Conversão para escala de cinza e binarização
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Encontrar contornos
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Cópia da imagem para desenho
img_retangulos = img.copy()

# Lista para guardar os pontos dos retângulos encontrados
pontos_retangulos = []

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 1000:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            cv2.drawContours(img_retangulos, [approx], -1, (0, 255, 0), 2)  # retângulo verde

            # Marca cada vértice com um círculo vermelho
            for idx, point in enumerate(approx):
                x, y = point[0]
                cv2.circle(img_retangulos, (x, y), 6, (0, 0, 255), -1)  # ponto vermelho
                cv2.putText(img_retangulos, f"P{idx+1}", (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                pontos_retangulos.append((x, y))

# Exibir imagem com retângulo e pontos
showImage(img_retangulos, "Retângulo com Pontos Marcados")
