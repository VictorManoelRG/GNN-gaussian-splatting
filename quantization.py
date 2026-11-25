import cv2
import numpy as np

def gerar_mascara_cor_bruta(caminho_imagem):
    # 1. Carregar a imagem
    img = cv2.imread(caminho_imagem)
    if img is None:
        print("Erro: Imagem não encontrada.")
        return

    # 2. Converter BGR (padrão do OpenCV) para HSV
    # O HSV separa a "Cor" (Hue) da "Intensidade de Luz" (Value), o que facilita isolar o urso rosa.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 3. Definir a faixa de cor do "Rosa/Magenta" no HSV
    # (O rosa do urso está entre o vermelho e o roxo. Aqui pegamos uma faixa ampla para capturar os tons claros e escuros)
    lower_pink = np.array([140, 50, 50])   # Tom mínimo (Hue 140 é magenta)
    upper_pink = np.array([175, 255, 255]) # Tom máximo

    # 4. Criar a máscara binária (Preto e Branco)
    # Pixels dentro da faixa viram 255 (Branco), pixels fora viram 0 (Preto).
    mascara = cv2.inRange(hsv, lower_pink, upper_pink)

    # (Opcional) Limpeza da máscara: remove pequenos ruídos e buracos
    kernel = np.ones((5,5), np.uint8)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)

    # 5. Criar a imagem final com a cor 100% BRUTA (pura)
    # Criamos uma imagem preta do mesmo tamanho
    imagem_cor_bruta = np.zeros_like(img)

    # Definimos a cor RGB bruta que queremos. Ex: Rosa Magenta Puro (255, 0, 255)
    # ATENÇÃO: O OpenCV usa BGR, então invertemos a ordem: (Azul, Verde, Vermelho)
    cor_rgb_bruta = (255, 0, 255) 

    # Aplicamos a cor pura apenas onde a máscara é branca
    imagem_cor_bruta[mascara == 255] = cor_rgb_bruta

    # 6. Exibir resultados
    cv2.imshow("Imagem Original", img)
    cv2.imshow("Mascara Binaria (Preto e Branco)", mascara)
    cv2.imshow("Mascara com Cor RGB Bruta", imagem_cor_bruta)
    
    # Salvar em arquivo (opcional)
    cv2.imwrite('mascara_cor_bruta_urso.png', imagem_cor_bruta)
    cv2.imwrite('mascara_binaria.png', mascara)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Execute a função passando o caminho da sua imagem
gerar_mascara_cor_bruta('/home/victor/Documentos/gaussian_grouping/gaussian-grouping/exp_res/gat_8heads/gat/images/pydensecrf_results/frame_0002.png')