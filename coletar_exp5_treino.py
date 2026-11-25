#!/usr/bin/env python3
"""
Coleta os resultados do Experimento 5 (contribuição do treinamento da GNN,
cena Teatime recortada) e emite as tabelas LaTeX prontas para a tese.

As três configurações comparadas mudam APENAS o quanto a rede foi treinada:

  sem_rede -> HDBSCAN direto nas features de entrada (xyz normalizado + Lab),
              sem rede nenhuma
  epoca0   -> embedding da GAT com os pesos da inicialização, sem treino
  epoca5   -> embedding da GAT após 5 épocas
  epoca20  -> embedding da GAT após 20 épocas
  padrao   -> treino completo; os números vêm do Experimento 1, que rodou
              exatamente esta configuração (prática de reaproveitar a baseline
              em vez de re-executá-la)

Cada configuração tem 5 repetições; as tabelas reportam média ± desvio padrão,
no mesmo formato dos Experimentos 1 a 4. Só ARI, NMI e Silhouette Score entram
na tabela 3D, que é o conjunto adotado na tese.

Uso:
    python coletar_exp5_treino.py                      # tabelas no stdout
    python coletar_exp5_treino.py --out tabelas_exp5.tex
"""

import argparse
import glob
import json
import os
import re
import statistics

RAIZ = os.path.dirname(os.path.abspath(__file__))
EXP5 = os.path.join(RAIZ, "experimentos_finais", "experimento5")
EXP1 = os.path.join(RAIZ, "experimentos_finais", "experimento1")

REPETICOES = [1, 2, 3, 4, 5]

# Rótulo exibido na tabela -> (prefixo das rodadas, onde estão os arquivos)
CONFIGS = [
    ("HDBSCAN nas features de entrada", "sem_rede", "exp5"),
    ("GAT sem treino (época 0)",        "epoca0",   "exp5"),
    ("GAT com 5 épocas",                "epoca5",   "exp5"),
    ("GAT com 20 épocas",               "epoca20",  "exp5"),
    ("Padrão (treino completo)",        "padrao",   "exp1"),
]

LINHA_METRICA = re.compile(r"^\s{2}(?P<chave>\S.*?)\s*:\s*(?P<valor>.+?)\s*$")

CHAVES_3D = {
    "ARI (Adjusted Rand Index)": "ari",
    "NMI (Normalized Mutual Info)": "nmi",
    "Silhouette Score": "silhouette",
    "Clusters Finais (HDBSCAN)": "clusters",
    "Épocas Treinadas": "epocas",
}

# Linhas que o --diagnose_embedding imprime no log do treino.
RE_MARGEM = re.compile(r"Margem de separação\s*:\s*([+-][\d.]+)")
RE_INTRA = re.compile(r"Similaridade média INTRA-rótulo\s*:\s*([+-][\d.]+)")
RE_INTER = re.compile(r"Similaridade média INTER-rótulo\s*:\s*([+-][\d.]+)")


def br(valor, casas=4):
    """Formata um float no padrão da tese: vírgula decimal escapada em LaTeX."""
    if valor is None:
        return "--"
    return f"{valor:.{casas}f}".replace(".", "{,}")


def media_dp(valores, casas=4):
    """'média ± desvio padrão' em LaTeX, ou '--' quando não há dados."""
    vals = [v for v in valores if v is not None]
    if not vals:
        return "--"
    m = statistics.mean(vals)
    dp = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return f"{br(m, casas)} $\\pm$ {br(dp, casas)}"


def caminho_report(prefixo, rep, origem):
    """O Exp. 1 guarda os relatórios achatados em metricas/relatorios/."""
    if origem == "exp1":
        return os.path.join(EXP1, "metricas", "relatorios", f"{prefixo}_r{rep}.txt")
    padroes = sorted(glob.glob(
        os.path.join(EXP5, f"{prefixo}_r{rep}", "gat", "GAT_report_*.txt")))
    return padroes[-1] if padroes else None


def caminho_miou(prefixo, rep, origem):
    base = EXP1 if origem == "exp1" else EXP5
    return os.path.join(base, "metricas", f"{prefixo}_r{rep}_miou.json")


def ler_report(caminho):
    """Extrai as métricas 3D de um GAT_report_*.txt."""
    if not caminho or not os.path.exists(caminho):
        return None
    dados = {}
    with open(caminho, encoding="utf-8") as f:
        for linha in f:
            # O histórico de loss no fim do arquivo não segue este formato,
            # e cortar cedo evita varrer centenas de linhas à toa.
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


def ler_miou(caminho):
    """Extrai o bloco agregado das métricas 2D."""
    if not os.path.exists(caminho):
        return None
    with open(caminho, encoding="utf-8") as f:
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
    }


# O Experimento 1 não rodou com --diagnose_embedding, então a margem de
# separação da configuração Padrão vem das rodadas padraodiag_r*, feitas só
# para esse fim. As demais métricas dessa linha continuam vindo do Exp. 1.
PREFIXO_DIAGNOSTICO = {"padrao": "padraodiag"}


def ler_diagnostico(prefixo, rep):
    """Lê a margem de separação que o --diagnose_embedding imprime no log."""
    prefixo = PREFIXO_DIAGNOSTICO.get(prefixo, prefixo)
    caminho = os.path.join(EXP5, "logs", f"{prefixo}_r{rep}.log")
    if not os.path.exists(caminho):
        return {}
    texto = open(caminho, encoding="utf-8", errors="replace").read()
    out = {}
    for chave, regex in (("margem", RE_MARGEM), ("intra", RE_INTRA), ("inter", RE_INTER)):
        m = regex.search(texto)
        if m:
            out[chave] = float(m.group(1))
    return out


def coletar():
    """Devolve, por configuração, a lista de repetições com todas as métricas."""
    resultado = []
    for rotulo, prefixo, origem in CONFIGS:
        rodadas = []
        for rep in REPETICOES:
            r = {"rep": rep}
            rep3d = ler_report(caminho_report(prefixo, rep, origem))
            if rep3d is None:
                print(f"  ! {prefixo}_r{rep}: sem relatório — repetição ignorada")
                continue
            r.update(rep3d)

            m2d = ler_miou(caminho_miou(prefixo, rep, origem))
            if m2d is None:
                print(f"  ! {prefixo}_r{rep}: sem *_miou.json — métricas 2D vazias")
            else:
                r.update(m2d)

            r.update(ler_diagnostico(prefixo, rep))
            rodadas.append(r)
        resultado.append({"rotulo": rotulo, "prefixo": prefixo,
                          "origem": origem, "rodadas": rodadas})
    return resultado


def coluna(config, campo):
    return [r.get(campo) for r in config["rodadas"]]


def tabela_3d(configs):
    linhas = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Métricas de clusterização 3D em função do estágio de treinamento "
        r"da GNN (média $\pm$ desvio padrão de 5 execuções)}",
        r"\label{tab:metricas_3d_experimento5}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccc}",
        r"\hline",
        r"\textbf{Configuração} & \textbf{ARI} & \textbf{NMI} & "
        r"\textbf{Silhouette Score} & \textbf{Clusters Finais} \\",
        r"\hline",
    ]
    for c in configs:
        linhas.append(
            f"{c['rotulo']} & {media_dp(coluna(c, 'ari'))} & "
            f"{media_dp(coluna(c, 'nmi'))} & {media_dp(coluna(c, 'silhouette'))} & "
            f"{media_dp(coluna(c, 'clusters'), casas=1)} \\\\"
        )
    linhas += [r"\hline", r"\end{tabular}%", r"}", r"\end{table}"]
    return "\n".join(linhas)


def tabela_2d(configs):
    linhas = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Métricas de segmentação de instância 2D em função do estágio de "
        r"treinamento da GNN (IoU $\geq 0{,}5$; média $\pm$ desvio padrão de 5 execuções)}",
        r"\label{tab:metricas_2d_experimento5}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccccc}",
        r"\hline",
        r"\textbf{Configuração} & \textbf{mIoU} & \textbf{Precisão@0,5} & "
        r"\textbf{Recall@0,5} & \textbf{SQ@0,5} & \textbf{RQ@0,5} & \textbf{PQ@0,5} \\",
        r"\hline",
    ]
    for c in configs:
        linhas.append(
            f"{c['rotulo']} & {media_dp(coluna(c, 'miou'))} & "
            f"{media_dp(coluna(c, 'prec50'))} & {media_dp(coluna(c, 'rec50'))} & "
            f"{media_dp(coluna(c, 'sq50'))} & {media_dp(coluna(c, 'rq50'))} & "
            f"{media_dp(coluna(c, 'pq50'))} \\\\"
        )
    linhas += [r"\hline", r"\end{tabular}%", r"}", r"\end{table}"]
    return "\n".join(linhas)


def tabela_instancias(configs):
    gt = None
    for c in configs:
        for v in coluna(c, "inst_gt"):
            if v is not None:
                gt = v
                break
        if gt is not None:
            break
    linhas = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Número médio de instâncias por imagem, preditas em cada estágio de "
        r"treinamento, comparado ao \textit{ground truth} (média $\pm$ desvio padrão "
        r"de 5 execuções)}",
        r"\label{tab:instancias_experimento5}",
        r"\begin{tabular}{lc}",
        r"\hline",
        r"\textbf{Configuração} & \textbf{Instâncias preditas (média/imagem)} \\",
        r"\hline",
        f"\\textit{{Ground truth}} & {br(gt, 1) if gt is not None else '--'} \\\\",
        r"\hline",
    ]
    for c in configs:
        linhas.append(f"{c['rotulo']} & {media_dp(coluna(c, 'inst_pred'), casas=1)} \\\\")
    linhas += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(linhas)


def tabela_diagnostico(configs):
    """
    Separação do embedding medida diretamente, antes do HDBSCAN entrar: a
    similaridade cosseno média entre pares de mesmo rótulo DEVA menos a de
    pares de rótulos diferentes.
    """
    linhas = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Separação do espaço de \textit{embeddings} antes da clusterização: "
        r"similaridade cosseno média entre pares do mesmo rótulo e de rótulos "
        r"distintos (média $\pm$ desvio padrão de 5 execuções)}",
        r"\label{tab:separacao_experimento5}",
        r"\begin{tabular}{lccc}",
        r"\hline",
        r"\textbf{Configuração} & \textbf{Intra-rótulo} & \textbf{Inter-rótulo} & "
        r"\textbf{Margem} \\",
        r"\hline",
    ]
    for c in configs:
        linhas.append(
            f"{c['rotulo']} & {media_dp(coluna(c, 'intra'))} & "
            f"{media_dp(coluna(c, 'inter'))} & {media_dp(coluna(c, 'margem'))} \\\\"
        )
    linhas += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(linhas)


def resumo(configs):
    """Impressão legível no terminal, para conferir antes de colar na tese."""
    out = ["", "=" * 78, "RESUMO (média ± desvio padrão)", "=" * 78]
    for c in configs:
        n = len(c["rodadas"])
        epocas = [r.get("epocas") for r in c["rodadas"] if r.get("epocas") is not None]
        ep = f"{statistics.mean(epocas):.0f}" if epocas else "--"
        out.append(f"\n{c['rotulo']}  ({n} repetições, origem={c['origem']}, épocas={ep})")
        for campo, nome in (("ari", "ARI"), ("nmi", "NMI"),
                            ("silhouette", "Silhouette"), ("clusters", "Clusters"),
                            ("miou", "mIoU"), ("pq50", "PQ@0,5"),
                            ("inst_pred", "Instâncias/img"), ("margem", "Margem")):
            vals = [v for v in coluna(c, campo) if v is not None]
            if not vals:
                continue
            m = statistics.mean(vals)
            dp = statistics.stdev(vals) if len(vals) > 1 else 0.0
            out.append(f"    {nome:<16}: {m:.4f} ± {dp:.4f}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="Arquivo .tex de saída")
    args = ap.parse_args()

    configs = coletar()
    blocos = [
        tabela_3d(configs), "", tabela_2d(configs), "",
        tabela_instancias(configs), "", tabela_diagnostico(configs),
    ]
    texto = "\n".join(blocos)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(texto + "\n")
        print(f"✅ tabelas salvas em {args.out}")
    else:
        print(texto)
    print(resumo(configs))


if __name__ == "__main__":
    main()
