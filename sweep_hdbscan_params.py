#!/usr/bin/env python3
"""
Sweep de hiperparametros do HDBSCAN (--hdbscan_min_samples x
--hdbscan_cluster_selection_epsilon): roda o pipeline completo de
"train_gnn copy 3.py" uma vez por combinacao (mesma seed, so o HDBSCAN
muda), avalia cada resultado com calculate_miou_claude_full_table.py, e
reporta a combinacao com melhor metrica.

Cada combinacao reexecuta o pipeline inteiro (grafo + treino do GNN +
HDBSCAN + render), nao so o HDBSCAN isolado - reaproveitar embeddings
ja treinados exigiria alterar a estrutura do script principal, e isso
mantem o sweep simples e seguro de rodar sem mexer no que ja funciona.
Por isso o grid default e pequeno; use --dry_run pra ver os comandos e
medir quanto tempo uma combinacao leva antes de alargar o grid.

Exemplo:
    python sweep_hdbscan_params.py \\
        -m output/teatime --iteration 30000 \\
        --masks_path .../Annotations --deva_json .../pred.json \\
        --model_type_choice gat --gat_heads 8 \\
        --gt /home/victor/Downloads/lear_ovs/lerf_ovs/label/teatime
"""
import argparse
import itertools
import json
import os
import subprocess
import sys
import time


def build_train_cmd(args, min_samples, epsilon, exp_name):
    return [
        args.python_bin, "train_gnn copy 3.py",
        "-m", args.model_path,
        "--iteration", str(args.iteration),
        "--masks_path", args.masks_path,
        "--deva_json", args.deva_json,
        "--model_type_choice", args.model_type_choice,
        "--gat_heads", str(args.gat_heads),
        "--experiment_name", exp_name,
        "--output", args.output,
        "--hdbscan_min_samples", str(min_samples),
        "--hdbscan_cluster_selection_epsilon", str(epsilon),
        "--seed", str(args.seed),
        "--no_visualize_3d",
    ]


def build_eval_cmd(args, exp_name, out_json):
    pred_dir = os.path.join(args.output, exp_name, args.model_type_choice, "images_coverage")
    cmd = [
        args.python_bin, "calculate_miou_claude_full_table.py",
        "--pred", pred_dir,
        "--gt", args.gt,
        "--out", out_json,
    ]
    return cmd, pred_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-m", "--model_path", required=True)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--masks_path", required=True)
    parser.add_argument("--deva_json", required=True)
    parser.add_argument("--model_type_choice", default="gat")
    parser.add_argument("--gat_heads", type=int, default=8)
    parser.add_argument("--gt", required=True,
                         help="Diretorio com o ground truth (jsons de label) para avaliacao.")
    parser.add_argument("--output", default="exp_res")
    parser.add_argument("--experiment_prefix", default="sweep_hdbscan")
    parser.add_argument("--seed", type=int, default=42,
                         help="Mesma seed em todas as combinacoes, pra isolar o efeito do HDBSCAN.")
    parser.add_argument("--min_samples_grid", default="4,6,8",
                         help="Valores de --hdbscan_min_samples a testar, separados por virgula.")
    parser.add_argument("--epsilon_grid", default="0.0,0.1",
                         help="Valores de --hdbscan_cluster_selection_epsilon a testar, separados por virgula.")
    parser.add_argument("--rank_by", default="mean_pq_50",
                         help="Chave de 'aggregated' no relatorio do avaliador usada pra ranquear "
                              "as combinacoes (default: Panoptic Quality @ IoU>=0.5). Outras "
                              "opcoes uteis: mean_iou_all_matches, mean_f1_50, mean_f1_75.")
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument("--results_json", default="hdbscan_sweep_results.json")
    parser.add_argument("--dry_run", action="store_true",
                         help="So imprime os comandos que seriam executados, sem rodar nada.")
    args = parser.parse_args()

    min_samples_values = [int(v) for v in args.min_samples_grid.split(",")]
    epsilon_values = [float(v) for v in args.epsilon_grid.split(",")]
    combos = list(itertools.product(min_samples_values, epsilon_values))

    print(f"Sweep: {len(combos)} combinacoes | min_samples={min_samples_values} | "
          f"epsilon={epsilon_values}")

    if args.dry_run:
        for min_samples, epsilon in combos:
            exp_name = f"{args.experiment_prefix}_ms{min_samples}_eps{epsilon}"
            print("\n# " + exp_name)
            print(" ".join(build_train_cmd(args, min_samples, epsilon, exp_name)))
        return

    log_dir = os.path.join(args.output, f"{args.experiment_prefix}_logs")
    os.makedirs(log_dir, exist_ok=True)

    results = []
    sweep_start = time.time()

    for i, (min_samples, epsilon) in enumerate(combos, start=1):
        exp_name = f"{args.experiment_prefix}_ms{min_samples}_eps{epsilon}"
        print(f"\n{'=' * 80}\n[{i}/{len(combos)}] min_samples={min_samples}  epsilon={epsilon}  "
              f"-> {exp_name}\n{'=' * 80}")

        train_cmd = build_train_cmd(args, min_samples, epsilon, exp_name)
        train_log_path = os.path.join(log_dir, f"{exp_name}_train.log")
        t0 = time.time()
        with open(train_log_path, "w") as log_f:
            proc = subprocess.run(train_cmd, stdout=log_f, stderr=subprocess.STDOUT)
        train_time = time.time() - t0

        row = {"min_samples": min_samples, "epsilon": epsilon, "train_time_s": round(train_time, 1)}

        if proc.returncode != 0:
            print(f"  [FALHOU] treino (codigo {proc.returncode}) - log: {train_log_path}")
            row["status"] = "train_failed"
            results.append(row)
            continue

        eval_json_path = os.path.join(log_dir, f"{exp_name}_miou.json")
        eval_cmd, pred_dir = build_eval_cmd(args, exp_name, eval_json_path)
        eval_log_path = os.path.join(log_dir, f"{exp_name}_eval.log")
        with open(eval_log_path, "w") as log_f:
            proc = subprocess.run(eval_cmd, stdout=log_f, stderr=subprocess.STDOUT)

        if proc.returncode != 0 or not os.path.exists(eval_json_path):
            print(f"  [FALHOU] avaliacao (codigo {proc.returncode}) - log: {eval_log_path}")
            row["status"] = "eval_failed"
            results.append(row)
            continue

        with open(eval_json_path) as f:
            report = json.load(f)
        agg = report.get("aggregated", {})

        row["status"] = "ok"
        row["pred_dir"] = pred_dir
        row.update(agg)
        results.append(row)

        elapsed = time.time() - sweep_start
        avg_per_combo = elapsed / i
        eta_min = avg_per_combo * (len(combos) - i) / 60
        print(f"  mIoU={agg.get('mean_iou_all_matches', float('nan')):.4f}  "
              f"F1@0.5={agg.get('mean_f1_50', float('nan')):.4f}  "
              f"PQ@0.5={agg.get('mean_pq_50', float('nan')):.4f}  "
              f"({train_time / 60:.1f} min de treino, ETA restante ~{eta_min:.0f} min)")

    with open(args.results_json, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    ok_results = [r for r in results if r.get("status") == "ok"]
    if not ok_results:
        print("\nNenhuma combinacao terminou com sucesso — veja os logs em " + log_dir)
        return

    ok_results.sort(key=lambda r: r.get(args.rank_by, float("-inf")), reverse=True)

    print(f"\n{'=' * 80}\nRANKING (por {args.rank_by})\n{'=' * 80}")
    print(f"{'min_samples':>11} {'epsilon':>8} {'mIoU':>8} {'F1@0.5':>8} {'PQ@0.5':>8} {'F1@0.75':>8}")
    for r in ok_results:
        print(f"{r['min_samples']:>11} {r['epsilon']:>8} "
              f"{r.get('mean_iou_all_matches', float('nan')):>8.4f} "
              f"{r.get('mean_f1_50', float('nan')):>8.4f} "
              f"{r.get('mean_pq_50', float('nan')):>8.4f} "
              f"{r.get('mean_f1_75', float('nan')):>8.4f}")

    best = ok_results[0]
    print(f"\nMelhor combinacao encontrada:")
    print(f"  --hdbscan_min_samples {best['min_samples']} "
          f"--hdbscan_cluster_selection_epsilon {best['epsilon']}")
    print(f"Resultados completos (todas as combinacoes) salvos em: {args.results_json}")
    print(f"Logs de cada rodada em: {log_dir}")


if __name__ == "__main__":
    main()
