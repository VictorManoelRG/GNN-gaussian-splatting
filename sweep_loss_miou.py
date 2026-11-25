#!/usr/bin/env python3
"""
Sweep dos pesos da loss de `train_gnn copy 16.py` avaliado por mIoU/PQ 2D.

Por que existe: até agora o equilíbrio entre os termos da loss (margem
positiva, margem negativa, InfoNCE, repulsão de centróides) era escolhido no
olho, e comparado por ARI/NMI 3D — que medem concordância com os rótulos
DEVA/SAM, eles próprios ruidosos. A métrica que interessa para a tese é
mIoU/PQ contra o ground-truth do LERF-OVS. Este script fecha esse laço:
treina, renderiza SÓ os frames anotados (--render_frames, ~50x mais barato),
roda o avaliador 2D e junta tudo numa tabela.

Uso:
    python sweep_loss_miou.py --scene teatime
    python sweep_loss_miou.py --scene figurines --configs C1,C7
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time

RAIZ = os.path.dirname(os.path.abspath(__file__))
TREINO = os.path.join(RAIZ, "train_gnn copy 16.py")
AVALIADOR = os.path.join(RAIZ, "calculate_miou_claude_full_table.py")
DEVA_ROOT = os.path.join(RAIZ, "Tracking-Anything-with-DEVA/example/output_gaussian_dataset")
GT_ROOT = "/home/victor/Downloads/lear_ovs/lerf_ovs/label"

# Frames com ground-truth por cena (o resto das ~300 câmeras não é comparado
# com nada, então não é renderizado durante o sweep).
FRAMES_GT = {
    "teatime": "frame_00002,frame_00025,frame_00043,frame_00107,frame_00129,frame_00140",
    "figurines": "frame_00041,frame_00105,frame_00152,frame_00195",
    "ramen": None,
    "waldo_kitchen": None,
}

# Parâmetros de cena que não fazem parte do sweep (mantidos fixos para que a
# única variação entre linhas da tabela sejam os pesos da loss).
CENA_FIXA = {
    "teatime": ["--min_cluster_size", "500", "--hdbscan_min_samples", "3",
                "--target_gaussians", "200000"],
    "figurines": ["--min_cluster_size", "500", "--hdbscan_min_samples", "3",
                  "--target_gaussians", "200000"],
    "ramen": ["--min_cluster_size", "500", "--hdbscan_min_samples", "3",
              "--target_gaussians", "50000"],
    "waldo_kitchen": ["--min_cluster_size", "500", "--hdbscan_min_samples", "3",
                      "--target_gaussians", "200000"],
}

# Sweep coordenado em torno de um centro razoável (C1). Cada configuração muda
# UM eixo por vez em relação a C1, para que a diferença de mIoU seja atribuível
# a esse eixo. As três chaves de correção (detach/balanced) ficam ligadas em
# todas, já que a comparação anterior por ARI/NMI já as favoreceu.
BASE = dict(temperature=0.05, pos_margin=0.2, neg_margin=0.1,
            pos_w=1.0, neg_w=1.0, nce_w=1.0, rep=0.3)

CONFIGS = {
    "C1_base":        dict(BASE),
    "C2_rep0":        dict(BASE, rep=0.0),
    "C3_rep1":        dict(BASE, rep=1.0),
    "C4_nce2":        dict(BASE, nce_w=2.0),
    "C5_neg2":        dict(BASE, neg_w=2.0),
    "C6_pos2":        dict(BASE, pos_w=2.0),
    "C7_temp01":      dict(BASE, temperature=0.1),
    "C8_margens0":    dict(BASE, pos_margin=0.0, neg_margin=0.0),
    "C9_nce2_rep1":   dict(BASE, nce_w=2.0, rep=1.0),
    "C10_nce05":      dict(BASE, nce_w=0.5),

    # Etapa 2: mesmo eixo de pesos, mas com a clusterização num regime
    # sensato. Na etapa 1 todas as configs entregaram 8-14 clusters contra
    # 62 instâncias de GT, então o mIoU media sub-segmentação, não a loss.
    "D1_mcs200_rep1":  dict(BASE, rep=1.0, mcs=200),
    "D2_mcs100_rep1":  dict(BASE, rep=1.0, mcs=100),
    "D3_mcs100_rep03": dict(BASE, rep=0.3, mcs=100),
    "D4_mcs100_nce2":  dict(BASE, rep=1.0, nce_w=2.0, mcs=100),
    "D5_mcs50_rep1":   dict(BASE, rep=1.0, mcs=50),
    "D6_mcs100_neg2":  dict(BASE, rep=1.0, neg_w=2.0, mcs=100),

    # O eixo do termo negativo foi o mais forte da etapa 1 (C5: neg_w=2.0
    # liderou mIoU E PQ50 simultaneamente), coerente com o diagnóstico de
    # sub-segmentação: é o termo que separa vizinhos de rótulos diferentes.
    # Estas testam até onde esse eixo continua pagando.
    "D7_neg4":         dict(BASE, rep=1.0, neg_w=4.0, mcs=100),
    "D8_neg2_nce2":    dict(BASE, rep=1.0, neg_w=2.0, nce_w=2.0, mcs=100),
    "D9_neg4_mcs500":  dict(BASE, rep=1.0, neg_w=4.0),
    "D10_neg8":        dict(BASE, rep=1.0, neg_w=8.0, mcs=100),

    # Temperatura foi o outro eixo forte da etapa 1 (C7: temp=0.1 deu o melhor
    # PQ50), e na direção contra-intuitiva — softmax MAIS suave, não mais
    # afiado. Estas cruzam esse eixo com o vencedor de mIoU (neg_w=2).
    "D11_temp01_neg2": dict(BASE, temperature=0.1, neg_w=2.0, rep=1.0, mcs=100),
    "D12_temp02":      dict(BASE, temperature=0.2, neg_w=2.0, rep=1.0, mcs=100),
    "D13_temp01_m500": dict(BASE, temperature=0.1, neg_w=2.0, rep=1.0),

    # C9 (nce_w=2 + rep=1) foi super-aditiva: 0.3554 de mIoU contra 0.2077 e
    # 0.2365 dos dois eixos isolados. Faz sentido — reforçar a classificação
    # contra centróides só paga se os centróides estiverem separados, que é o
    # que a repulsão garante. Estas exploram a vizinhança desse ponto.
    "E1_nce4_rep1":    dict(BASE, nce_w=4.0, rep=1.0),
    "E2_nce2_rep2":    dict(BASE, nce_w=2.0, rep=2.0),
    "E3_nce4_rep2":    dict(BASE, nce_w=4.0, rep=2.0),
    "E4_C9_neg2":      dict(BASE, nce_w=2.0, rep=1.0, neg_w=2.0),
    "E5_C9_temp01":    dict(BASE, nce_w=2.0, rep=1.0, temperature=0.1),
    "E6_C9_mcs100":    dict(BASE, nce_w=2.0, rep=1.0, mcs=100),

    # As duas melhores configs até aqui compartilham a razão nce_w:rep = 2:1
    # (C9 = 2:1 -> 0.3554; E3 = 4:2 -> 0.3860), enquanto as fora dessa razão
    # caem (E1 = 4:1 -> 0.2717; E2 = 2:2 -> 0.3499). Escalar os dois mantendo
    # a razão subiu o mIoU, então estas testam até onde isso vai. Se F1/F2
    # continuarem subindo, o que importa não é o valor absoluto de nenhum dos
    # dois e sim o peso conjunto do bloco de centróides contra as margens.
    "F1_nce8_rep4":    dict(BASE, nce_w=8.0, rep=4.0),
    "F2_nce16_rep8":   dict(BASE, nce_w=16.0, rep=8.0),
    "F3_nce6_rep3":    dict(BASE, nce_w=6.0, rep=3.0),

    # Estabilidade: o colapso isolado do F1 (8:4) mostrou que a paisagem é
    # não-monotônica, então diferenças de ~0.03 entre configs boas podem ser
    # ruído de inicialização. Estas repetem o MESMO ponto com sementes
    # diferentes para medir a dispersão e saber quanto de diferença entre
    # configs é sinal de verdade.
    "S1_C9_seed1":     dict(BASE, nce_w=2.0, rep=1.0, seed=1),
    "S2_C9_seed7":     dict(BASE, nce_w=2.0, rep=1.0, seed=7),
    "S3_C9_seed123":   dict(BASE, nce_w=2.0, rep=1.0, seed=123),
    "S4_E3_seed1":     dict(BASE, nce_w=4.0, rep=2.0, seed=1),
    "S5_E3_seed7":     dict(BASE, nce_w=4.0, rep=2.0, seed=7),
    "S6_F1_seed1":     dict(BASE, nce_w=8.0, rep=4.0, seed=1),

    # E6 (nce 2:1 + min_cluster_size 100) foi o único ponto igualmente bom nas
    # duas cenas (teatime 0.4403 / figurines 0.4343), então é ele que precisa
    # de verificação de dispersão antes de virar recomendação.
    "G1_E6_seed1":     dict(BASE, nce_w=2.0, rep=1.0, mcs=100, seed=1),
    "G2_E6_seed7":     dict(BASE, nce_w=2.0, rep=1.0, mcs=100, seed=7),
    "G3_E6_pesos1":    dict(BASE, mcs=100),

    # Ablações que respondem "a loss/GNN está fazendo alguma coisa?". H1 usa a
    # rede recém-inicializada, sem treino nenhum; H2 descarta a rede e
    # clusteriza as features de entrada. Se qualquer uma empatar com o
    # pipeline treinado (~0.43-0.45 em teatime com mcs=100), a etapa de
    # aprendizado não está contribuindo.
    "H1_sem_treino":   dict(BASE, mcs=100, epochs=0),
    "H2_raw_features": dict(BASE, mcs=100, raw_features=True),
    "H3_treino20":     dict(BASE, mcs=100, epochs=20),

    # A curva 0/20/200 épocas (0.188 -> 0.243 -> 0.450) ainda subia forte no
    # limite do laço, e o early stopping (paciência 20) nunca disparou nessas
    # rodadas: o treino parou por limite fixo, não por convergência. Estas
    # testam se há ganho parado acima de 200.
    "I1_treino400":    dict(BASE, mcs=100, epochs=400),
    "I2_treino800":    dict(BASE, mcs=100, epochs=800),
}


def comando_treino(cfg, nome_exp, cena, out_dir, frames):
    # min_cluster_size pode ser sobrescrito por configuração: na etapa 1 do
    # sweep ele ficou fixo em 500 e TODAS as configs sub-segmentaram (8 a 14
    # clusters finais contra 62 instâncias no ground-truth), o que fazia o
    # mIoU refletir mais a clusterização do que os pesos da loss.
    fixos = [a for a in CENA_FIXA[cena]]
    if "mcs" in cfg:
        i = fixos.index("--min_cluster_size")
        fixos[i + 1] = str(cfg["mcs"])

    cmd = [
        sys.executable, TREINO,
        "-m", f"output/{cena}",
        "--iteration", "30000",
        "--masks_path", f"{DEVA_ROOT}/{cena}/Annotations",
        "--model_type_choice", "gat", "--gat_heads", "8",
        "--loss_temperature", str(cfg["temperature"]),
        "--loss_pos_margin", str(cfg["pos_margin"]),
        "--loss_neg_margin", str(cfg["neg_margin"]),
        "--loss_pos_weight", str(cfg["pos_w"]),
        "--loss_neg_weight", str(cfg["neg_w"]),
        "--loss_nce_weight", str(cfg["nce_w"]),
        "--loss_color_weight", "0",
        "--loss_centroid_repulsion_weight", str(cfg["rep"]),
        "--loss_detach_centroids", "--loss_nce_class_balanced",
        "--seed", str(cfg.get("seed", 42)),
        "--train_epochs", str(cfg.get("epochs", 200)),
        "--experiment_name", nome_exp,
        "--output", out_dir,
        "--no_visualize_3d", "--no_render_pca",
        "--no_bbox", "--no_render_full_pointcloud",
    ] + fixos
    if cfg.get("raw_features"):
        cmd += ["--cluster_on_raw_features"]
    if frames:
        cmd += ["--render_frames", frames]
    return cmd


def avalia_miou(pred_dir, gt_dir, saida_json):
    cmd = [sys.executable, AVALIADOR, "--pred", pred_dir, "--gt", gt_dir,
           "--out", saida_json, "--quiet"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if not os.path.exists(saida_json):
        return None, r.stdout[-2000:] + r.stderr[-2000:]
    with open(saida_json, encoding="utf-8") as f:
        dados = json.load(f)
    return dados.get("aggregated"), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="teatime")
    ap.add_argument("--configs", default=None,
                    help="Subconjunto de configs, separadas por vírgula (default: todas)")
    ap.add_argument("--out_root", default="exp_res_sweep_loss")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    cena = args.scene
    nomes = list(CONFIGS) if not args.configs else [c.strip() for c in args.configs.split(",")]
    gt_dir = os.path.join(GT_ROOT, cena)
    out_root = os.path.join(RAIZ, args.out_root)
    os.makedirs(out_root, exist_ok=True)
    csv_path = args.csv or os.path.join(out_root, f"resultados_{cena}.csv")

    campos = ["config", "cena", "temperature", "pos_margin", "neg_margin",
              "pos_w", "neg_w", "nce_w", "rep", "mcs", "seed", "epochs", "raw",
              "mIoU", "PQ50", "SQ50", "RQ50", "PQ75", "F1_50",
              "gt_inst", "pred_inst", "segundos"]
    novo = not os.path.exists(csv_path)
    f_csv = open(csv_path, "a", newline="", encoding="utf-8")
    escritor = csv.DictWriter(f_csv, fieldnames=campos)
    if novo:
        escritor.writeheader()
        f_csv.flush()

    for nome in nomes:
        cfg = CONFIGS[nome]
        nome_exp = f"{cena}_{nome}"
        t0 = time.time()
        print(f"\n{'='*70}\n▶ {nome_exp}  {cfg}\n{'='*70}", flush=True)

        log_path = os.path.join(out_root, f"{nome_exp}.log")
        with open(log_path, "w", encoding="utf-8") as logf:
            r = subprocess.run(comando_treino(cfg, nome_exp, cena, out_root,
                                              FRAMES_GT.get(cena)),
                               cwd=RAIZ, stdout=logf, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            print(f"❌ treino falhou (rc={r.returncode}); ver {log_path}", flush=True)
            continue

        pred_dir = os.path.join(out_root, nome_exp, "gat", "images_coverage")
        agg, erro = avalia_miou(pred_dir, gt_dir,
                                os.path.join(out_root, f"{nome_exp}_miou.json"))
        dt = time.time() - t0
        if agg is None:
            print(f"❌ avaliação falhou: {erro}", flush=True)
            continue

        linha = {
            "config": nome, "cena": cena,
            "temperature": cfg["temperature"], "pos_margin": cfg["pos_margin"],
            "neg_margin": cfg["neg_margin"], "pos_w": cfg["pos_w"],
            "neg_w": cfg["neg_w"], "nce_w": cfg["nce_w"], "rep": cfg["rep"],
            "mcs": cfg.get("mcs", 500), "seed": cfg.get("seed", 42),
            "epochs": cfg.get("epochs", 200), "raw": int(bool(cfg.get("raw_features"))),
            "mIoU": round(agg.get("mean_iou_all_matches", 0), 4),
            "PQ50": round(agg.get("mean_pq_50", 0), 4),
            "SQ50": round(agg.get("mean_sq_50", 0), 4),
            "RQ50": round(agg.get("mean_rq_50", 0), 4),
            "PQ75": round(agg.get("mean_pq_75", 0), 4),
            "F1_50": round(agg.get("mean_f1_50", 0), 4),
            "gt_inst": agg.get("total_gt_instances", 0),
            "pred_inst": agg.get("total_pred_instances", 0),
            "segundos": round(dt, 1),
        }
        escritor.writerow(linha)
        f_csv.flush()
        print(f"✅ {nome}: mIoU={linha['mIoU']} PQ50={linha['PQ50']} "
              f"({dt:.0f}s)", flush=True)

    f_csv.close()
    print(f"\n📊 Tabela: {csv_path}")


if __name__ == "__main__":
    main()
