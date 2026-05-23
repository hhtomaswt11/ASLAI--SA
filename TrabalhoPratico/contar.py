import os

# Mete aqui o caminho para a tua pasta train
TRAIN_DIR = "new_fusion/"

# Extensões que contam como imagem
EXTENSOES_IMAGEM = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")

total_geral = 0

print("Número de imagens por pasta:\n")

for pasta in sorted(os.listdir(TRAIN_DIR)):
    caminho_pasta = os.path.join(TRAIN_DIR, pasta)

    if os.path.isdir(caminho_pasta):
        total = 0

        for ficheiro in os.listdir(caminho_pasta):
            if ficheiro.lower().endswith(EXTENSOES_IMAGEM):
                total += 1

        total_geral += total
        print(f"{pasta}: {total} imagens")

print("\nTotal geral:", total_geral, "imagens")
