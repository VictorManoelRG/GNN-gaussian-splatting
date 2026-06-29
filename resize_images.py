import os
import cv2
import argparse

def resize_images(directory_path, scale_factor):
    # Nome do subdiretório de saída
    output_dir = os.path.join(
        directory_path,
        f"resized_x{scale_factor}"
    )

    os.makedirs(output_dir, exist_ok=True)

    # Extensões aceitas
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")

    for filename in os.listdir(directory_path):

        if not filename.lower().endswith(valid_extensions):
            continue

        input_path = os.path.join(directory_path, filename)

        # Lê imagem
        image = cv2.imread(input_path)

        if image is None:
            print(f"Erro ao ler: {filename}")
            continue

        height, width = image.shape[:2]

        # Novo tamanho
        new_width = width // scale_factor
        new_height = height // scale_factor

        # Evita dimensões inválidas
        if new_width <= 0 or new_height <= 0:
            print(f"Imagem muito pequena para resize: {filename}")
            continue

        # Resize
        resized = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )

        # Caminho de saída
        output_path = os.path.join(output_dir, filename)

        # Salva imagem
        cv2.imwrite(output_path, resized)

        print(f"Salvo: {output_path}")

    print("\nConcluído!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Resize de imagens em um diretório."
    )

    parser.add_argument(
        "directory",
        type=str,
        help="Diretório contendo as imagens"
    )

    parser.add_argument(
        "scale_factor",
        type=int,
        help="Fator de redução (2 = divide por 2, 4 = divide por 4, etc)"
    )

    args = parser.parse_args()

    resize_images(args.directory, args.scale_factor)