from PIL import Image
import os

def converter_para_jpg(caminho_arquivo):
    # Abre a imagem
    imagem = Image.open(caminho_arquivo)
    
    # Garante que está em modo RGB para salvar JPG
    if imagem.mode in ("RGBA", "P"):
        imagem = imagem.convert("RGB")
    
    # Define novo caminho com extensão .jpg
    base, _ = os.path.splitext(caminho_arquivo)
    novo_caminho = base + ".jpg"
    
    # Salva a imagem em JPG
    imagem.save(novo_caminho, "JPEG")
    
    print(f"Imagem convertida e salva em: {novo_caminho}")

# Exemplo de uso:
converter_para_jpg(r"D:\img\18.webp")
