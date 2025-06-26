import numpy as np
import cv2
import os
from utils.imagem import mostrar
from utils.ocr import extrair_texto

# Caminho da imagem
CAMINHO_IMG = r"D:\img\20.jpg"

def encontrar_maior_contorno_util(img, limite_area=1000, limite_proporcao=(0.4, 2.5), limite_ocupacao=0.95):
    contornos = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos = contornos[0] if len(contornos) == 2 else contornos[1]
    
    altura_img, largura_img = img.shape
    area_img = altura_img * largura_img

    candidatos = []
    for cont in contornos:
        area = cv2.contourArea(cont)
        if area < limite_area or area > area_img * limite_ocupacao:
            continue

        x, y, w, h = cv2.boundingRect(cont)
        proporcao = w / float(h)
        if limite_proporcao[0] < proporcao < limite_proporcao[1]:
            peri = cv2.arcLength(cont, True)
            aprox = cv2.approxPolyDP(cont, 0.02 * peri, True)
            if len(aprox) == 4:
                candidatos.append((cont, area))

    if candidatos:
        candidatos = sorted(candidatos, key=lambda x: x[1], reverse=True)
        return candidatos[0][0]

    print("⚠️ Nenhum contorno adequado encontrado. Tentando o maior disponível...")
    if contornos:
        return max(contornos, key=cv2.contourArea)
    else:
        return None

def ordenar_pontos(pontos):
    pontos = pontos.reshape((4, 2))
    pontos_novos = np.zeros((4, 2), dtype=np.float32)

    soma = pontos.sum(axis=1)
    diff = np.diff(pontos, axis=1)

    pontos_novos[0] = pontos[np.argmin(soma)]  # topo esquerdo
    pontos_novos[2] = pontos[np.argmax(soma)]  # baixo direito
    pontos_novos[1] = pontos[np.argmin(diff)]  # topo direito
    pontos_novos[3] = pontos[np.argmax(diff)]  # baixo esquerdo

    return pontos_novos

def transformar_imagem(caminho_imagem):
    img = cv2.imread(caminho_imagem)
    if img is None:
        raise FileNotFoundError(f"❌ Imagem não encontrada: {caminho_imagem}")

    original = img.copy()
    (H, W) = img.shape[:2]

    imagem_cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imagem_cinza = cv2.equalizeHist(imagem_cinza)

    blur = cv2.GaussianBlur(imagem_cinza, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 120)
    mostrar(edges)

    contorno = encontrar_maior_contorno_util(edges)
    if contorno is None:
        raise ValueError("❌ Contorno muito pequeno ou não encontrado.")

    peri = cv2.arcLength(contorno, True)
    aprox = cv2.approxPolyDP(contorno, 0.02 * peri, True)

    if len(aprox) == 4:
        maior = aprox
    else:
        rect = cv2.minAreaRect(contorno)
        box = cv2.boxPoints(rect)
        maior = np.int0(box)

    # Desenhar e mostrar contorno
    cv2.drawContours(img, [maior], -1, (0, 255, 0), 2)
    mostrar(img)

    # Aplicar transformação de perspectiva alinhada
    pontos = ordenar_pontos(maior)
    largura = int(max(
        np.linalg.norm(pontos[0] - pontos[1]),
        np.linalg.norm(pontos[2] - pontos[3])
    ))
    altura = int(max(
        np.linalg.norm(pontos[0] - pontos[3]),
        np.linalg.norm(pontos[1] - pontos[2])
    ))

    pts1 = pontos.astype("float32")
    pts2 = np.array([[0, 0], [largura - 1, 0], [largura - 1, altura - 1], [0, altura - 1]], dtype="float32")
    matriz = cv2.getPerspectiveTransform(pts1, pts2)
    transformada = cv2.warpPerspective(original, matriz, (largura, altura))

    return transformada

def processar_para_ocr(img):
    img = cv2.resize(img, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 11, 9)
    return img

if __name__ == "__main__":
    imagem = transformar_imagem(CAMINHO_IMG)
    mostrar(imagem)
    img_final = processar_para_ocr(imagem)
    mostrar(img_final)
    texto = extrair_texto(img_final)
    print("Texto extraído:")
    print(texto)
