#!/usr/bin/env python3
"""
Monta a figura de comparação do Experimento 5 (contribuição do treinamento da
GNN), no mesmo formato das figuras dos experimentos anteriores da tese.

Os estágios disponíveis vão de "nenhuma rede" até o treino completo. Como nem
todos precisam entrar na figura da tese, quais aparecem é escolhido por
argumento:

    python gerar_figuras_exp5.py                          # todos, para inspeção
    python gerar_figuras_exp5.py --estagios sem_rede,epoca0,padrao --saida figuras/exp5_treino_comparacao.png

O frame é o mesmo usado nas demais figuras da cena Teatime (frame_00002), que
faz parte do ground truth anotado do LERF-OVS.
"""

import argparse
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

RAIZ = os.path.dirname(os.path.abspath(__file__))
EXP5 = os.path.join(RAIZ, "experimentos_finais", "experimento5")
EXP1 = os.path.join(RAIZ, "experimentos_finais", "experimento1")
FRAME = "frame_00002"
ORIGINAL = f"/home/victor/Downloads/lear_ovs/lerf_ovs/label/teatime/{FRAME}.jpg"

# chave -> (título no painel, pasta da rodada preservada)
# A r1 é a repetição preservada das rodadas novas; no Exp. 1 a pasta `padrao`
# já guarda a repetição que foi mantida em disco.
ESTAGIOS = {
    "sem_rede": ("HDBSCAN nas features de entrada",
                 os.path.join(EXP5, "sem_rede_r1", "gat")),
    "epoca0":   ("GAT sem treino (época 0)",
                 os.path.join(EXP5, "epoca0_r1", "gat")),
    "epoca5":   ("GAT com 5 épocas",
                 os.path.join(EXP5, "epoca5_r1", "gat")),
    "epoca20":  ("GAT com 20 épocas",
                 os.path.join(EXP5, "epoca20_r1", "gat")),
    "padrao":   ("Padrão (treino completo)",
                 os.path.join(EXP1, "padrao", "gat")),
}


def monta(paineis, destino, colunas):
    """Grade de imagens com título em cada painel, eixos desligados."""
    faltando = [c for _, c in paineis if not os.path.exists(c)]
    if faltando:
        for c in faltando:
            print(f"  ! ausente: {c}")
        return False

    linhas = (len(paineis) + colunas - 1) // colunas
    # 8x4.5 por painel mantém a proporção das figuras dos outros experimentos.
    fig, axes = plt.subplots(linhas, colunas, figsize=(8 * colunas, 4.6 * linhas))
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]

    for ax, (titulo, caminho) in zip(axes, paineis):
        ax.imshow(mpimg.imread(caminho))
        ax.set_title(titulo, fontsize=15)
        ax.axis("off")
    # Sobras da grade não devem aparecer como eixo vazio.
    for ax in axes[len(paineis):]:
        ax.axis("off")

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(destino)), exist_ok=True)
    fig.savefig(destino, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ {destino}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--estagios", default=",".join(ESTAGIOS),
                    help="Estágios a incluir, separados por vírgula, na ordem desejada. "
                         f"Disponíveis: {', '.join(ESTAGIOS)}")
    ap.add_argument("--saida", default=os.path.join(RAIZ, "figuras",
                                                    "exp5_treino_comparacao.png"))
    ap.add_argument("--colunas", type=int, default=2)
    ap.add_argument("--sem-original", action="store_true",
                    help="Omite o painel da imagem original.")
    args = ap.parse_args()

    paineis = [] if args.sem_original else [("Imagem original", ORIGINAL)]
    for chave in [e.strip() for e in args.estagios.split(",") if e.strip()]:
        if chave not in ESTAGIOS:
            raise SystemExit(f"estágio desconhecido: {chave}")
        titulo, base = ESTAGIOS[chave]
        paineis.append((titulo, os.path.join(base, "images_coverage", f"{FRAME}.png")))

    monta(paineis, args.saida, colunas=args.colunas)


if __name__ == "__main__":
    main()
