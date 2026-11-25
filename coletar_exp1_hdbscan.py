#!/usr/bin/env python3
"""
Coleta os resultados do sweep do Experimento 1 (sensibilidade do HDBSCAN,
dataset Teatime) e emite as tabelas LaTeX prontas para colar na tese.

Lê, para cada rodada `hdb_mcs<MCS>_ms<MS>` de exp_res_exp1_hdbscan/:
  - o GAT_report_*.txt mais recente     -> ARI, NMI, Silhouette, clusters finais,
                                           tempo de HDBSCAN
  - o <nome>_miou.json                  -> mIoU, precisão/recall@0,5, SQ/RQ/PQ@0,5
                                           e instâncias preditas por imagem

Só ARI, NMI e Silhouette Score entram nas tabelas 3D, que é o conjunto de
métricas de clusterização 3D adotado na tese.

Uso:
    python coletar_exp1_hdbscan.py                       # tabelas no stdout
    python coletar_exp1_hdbscan.py --out tabelas_exp1.tex
"""

import argparse
import glob
import json
import os
import re

RAIZ = os.path.dirname(os.path.abspath(__file__))
PADRAO_DIR = os.path.join(RAIZ, "exp_res_exp1_hdbscan")

# "  ARI (Adjusted Rand Index)               : 0.2690"
LINHA_METRICA = re.compile(r"^\s{2}(?P<chave>\S.*?)\s*:\s*(?P<valor>.+?)\s*$")

CHAVES_3D = {
    "ARI (Adjusted Rand Index)": "ari",
    "NMI (Normalized Mutual Info)": "nmi",
    "Silhouette Score": "silhouette",
    "Clusters Finais (HDBSCAN)": "clusters",
    "Pontos Válidos": "pontos",
    "Tempo HDBSCAN (segundos)": "tempo_hdbscan",
}


def br(valor, casas=4):
    """Formata um float no padrão da tese: vírgula decimal escapada em LaTeX."""
    if valor is None:
        return "--"
    return f"{valor:.{casas}f}".replace(".", "{,}")


def ler_report(dir_rodada):
    """Extrai as métricas 3D do GAT_report_*.txt mais recente da rodada."""
    reports = sorted(glob.glob(os.path.join(dir_rodada, "gat", "GAT_report_*.txt")))
    if not reports:
        return None
    caminho = reports[-1]

    dados = {"report": os.path.basename(caminho)}
    with open(caminho, encoding="utf-8") as f:
        for linha in f:
            # O histórico de loss no fim do arquivo não tem ':' nesse formato,
            # mas cortar cedo evita varrer centenas de linhas à toa.
            if linha.startswith("HISTÓRICO DE LOSS"):
                break
            m = LINHA_METRICA.match(linha.rstrip("\n"))
            if not m:
                continue
            campo = CHAVES_3D.get(m.group("chave"))
            if campo is None:
                continue
            try:
                dados[campo] = float(m.group("valor"))
            except ValueError:
                pass
    return dados


def ler_miou(caminho_json):
    """Extrai o bloco agregado das métricas 2D."""
    if not os.path.exists(caminho_json):
        return None
    with open(caminho_json, encoding="utf-8") as f:
        ag = json.load(f).get("aggregated", {})
    return {
        "miou": ag.get("mean_iou_all_matches"),
        "prec50": ag.get("mean_precision_50"),
        "rec50": ag.get("mean_recall_50"),
        "sq50": ag.get("mean_sq_50"),
        "rq50": ag.get("mean_rq_50"),
        "pq50": ag.get("mean_pq_50"),
        "inst_pred": ag.get("mean_pred_instances_per_image"),
        "inst_gt": ag.get("mean_gt_instances_per_image"),
        "n_imagens": ag.get("total_images"),
    }


def coletar(base):
    """Varre exp_res_exp1_hdbscan/ e devolve as rodadas ordenadas por (mcs, ms)."""
    rodadas = []
    for dir_rodada in sorted(glob.glob(os.path.join(base, "hdb_mcs*_ms*"))):
        nome = os.path.basename(dir_rodada)
        m = re.match(r"hdb_mcs(\d+)_ms(\d+)$", nome)
        if not m:
            continue
        r = {"nome": nome, "mcs": int(m.group(1)), "ms": int(m.group(2))}

        rep = ler_report(dir_rodada)
        if rep is None:
            print(f"  ! {nome}: sem GAT_report_*.txt — rodada incompleta, ignorada")
            continue
        r.update(rep)

        miou = ler_miou(os.path.join(base, f"{nome}_miou.json"))
        if miou is None:
            print(f"  ! {nome}: sem {nome}_miou.json — métricas 2D ficarão vazias")
        else:
            r.update(miou)

        rodadas.append(r)

    rodadas.sort(key=lambda r: (r["mcs"], r["ms"]))
    return rodadas


def melhor(rodadas, campo):
    """Índice da rodada com maior valor do campo (para negritar na tabela)."""
    validos = [r for r in rodadas if r.get(campo) is not None]
    if not validos:
        return None
    return max(validos, key=lambda r: r[campo])["nome"]


def negrito(texto, destacar):
    return f"\\textbf{{{texto}}}" if destacar else texto


def tabela_3d(rodadas):
    top = {c: melhor(rodadas, c) for c in ("ari", "nmi", "silhouette")}
    linhas = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Métricas de clusterização 3D para cada configuração do HDBSCAN "
        r"na cena \textit{Teatime}. Melhor valor de cada métrica em negrito.}",
        r"\label{tab:exp1_metricas_3d}",
        r"\begin{tabular}{cccccc}",
        r"\hline",
        r"\textbf{min\_cluster\_size} & \textbf{min\_samples} & \textbf{ARI} & "
        r"\textbf{NMI} & \textbf{Silhouette Score} & \textbf{Clusters Finais} \\",
        r"\hline",
    ]
    mcs_anterior = None
    for r in rodadas:
        if mcs_anterior is not None and r["mcs"] != mcs_anterior:
            linhas.append(r"\hline")
        mcs_anterior = r["mcs"]
        clusters = r.get("clusters")
        linhas.append(
            f"{r['mcs']} & {r['ms']} & "
            f"{negrito(br(r.get('ari')), top['ari'] == r['nome'])} & "
            f"{negrito(br(r.get('nmi')), top['nmi'] == r['nome'])} & "
            f"{negrito(br(r.get('silhouette')), top['silhouette'] == r['nome'])} & "
            f"{int(clusters) if clusters is not None else '--'} \\\\"
        )
    linhas += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(linhas)


def tabela_2d(rodadas):
    top = {c: melhor(rodadas, c) for c in ("miou", "pq50")}
    linhas = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Métricas de segmentação de instância 2D para cada configuração do "
        r"HDBSCAN na cena \textit{Teatime} (IoU $\geq 0{,}5$). Melhores mIoU e PQ em negrito.}",
        r"\label{tab:exp1_metricas_2d}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{cccccccc}",
        r"\hline",
        r"\textbf{min\_cluster\_size} & \textbf{min\_samples} & \textbf{mIoU} & "
        r"\textbf{Precisão@0,5} & \textbf{Recall@0,5} & \textbf{SQ@0,5} & "
        r"\textbf{RQ@0,5} & \textbf{PQ@0,5} \\",
        r"\hline",
    ]
    mcs_anterior = None
    for r in rodadas:
        if mcs_anterior is not None and r["mcs"] != mcs_anterior:
            linhas.append(r"\hline")
        mcs_anterior = r["mcs"]
        linhas.append(
            f"{r['mcs']} & {r['ms']} & "
            f"{negrito(br(r.get('miou')), top['miou'] == r['nome'])} & "
            f"{br(r.get('prec50'))} & {br(r.get('rec50'))} & "
            f"{br(r.get('sq50'))} & {br(r.get('rq50'))} & "
            f"{negrito(br(r.get('pq50')), top['pq50'] == r['nome'])} \\\\"
        )
    linhas += [r"\hline", r"\end{tabular}%", r"}", r"\end{table}"]
    return "\n".join(linhas)


def tabela_instancias(rodadas):
    gt = next((r.get("inst_gt") for r in rodadas if r.get("inst_gt") is not None), None)
    linhas = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Número médio de instâncias por imagem preditas por cada configuração "
        r"do HDBSCAN, comparado ao \textit{ground truth} da cena \textit{Teatime}.}",
        r"\label{tab:exp1_instancias}",
        r"\begin{tabular}{ccc}",
        r"\hline",
        r"\textbf{min\_cluster\_size} & \textbf{min\_samples} & "
        r"\textbf{Instâncias preditas (média/imagem)} \\",
        r"\hline",
        f"\\multicolumn{{2}}{{c}}{{\\textit{{Ground truth}}}} & {br(gt, 1)} \\\\",
        r"\hline",
    ]
    mcs_anterior = None
    for r in rodadas:
        if mcs_anterior is not None and r["mcs"] != mcs_anterior:
            linhas.append(r"\hline")
        mcs_anterior = r["mcs"]
        linhas.append(f"{r['mcs']} & {r['ms']} & {br(r.get('inst_pred'), 1)} \\\\")
    linhas += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(linhas)


def resumo(rodadas):
    """Panorama em texto puro, para escrever a discussão sem reler as tabelas."""
    larg = "{:>17} {:>12} {:>8} {:>8} {:>11} {:>9} {:>8} {:>8} {:>9}"
    linhas = [
        larg.format("min_cluster_size", "min_samples", "ARI", "NMI", "Silhouette",
                    "Clusters", "mIoU", "PQ@0,5", "Inst/img"),
        "-" * 100,
    ]
    for r in rodadas:
        def v(c, casas=4):
            x = r.get(c)
            return f"{x:.{casas}f}" if x is not None else "--"
        clusters = r.get("clusters")
        linhas.append(larg.format(
            r["mcs"], r["ms"], v("ari"), v("nmi"), v("silhouette"),
            int(clusters) if clusters is not None else "--",
            v("miou"), v("pq50"), v("inst_pred", 1)))

    for campo, rotulo in (("ari", "ARI"), ("miou", "mIoU"), ("pq50", "PQ@0,5")):
        nome = melhor(rodadas, campo)
        if nome:
            r = next(x for x in rodadas if x["nome"] == nome)
            linhas.append(f"Melhor {rotulo}: min_cluster_size={r['mcs']}, "
                          f"min_samples={r['ms']} ({r[campo]:.4f})")
    return "\n".join(linhas)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=PADRAO_DIR,
                    help="Diretório de saída do sweep (default: exp_res_exp1_hdbscan)")
    ap.add_argument("--out", default=None,
                    help="Grava as tabelas LaTeX neste arquivo além de imprimi-las")
    args = ap.parse_args()

    if not os.path.isdir(args.dir):
        raise SystemExit(f"Diretório não encontrado: {args.dir}\n"
                         f"Rode antes: bash run_exp1_hdbscan_teatime.sh")

    rodadas = coletar(args.dir)
    if not rodadas:
        raise SystemExit(f"Nenhuma rodada encontrada em {args.dir}")

    print(f"\n{len(rodadas)} rodada(s) coletada(s) de {args.dir}\n")
    print(resumo(rodadas))

    tex = "\n\n".join([tabela_3d(rodadas), tabela_2d(rodadas), tabela_instancias(rodadas)])
    print("\n" + "=" * 100 + "\nTABELAS LATEX\n" + "=" * 100 + "\n")
    print(tex)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(tex + "\n")
        print(f"\nTabelas gravadas em: {args.out}")


if __name__ == "__main__":
    main()
