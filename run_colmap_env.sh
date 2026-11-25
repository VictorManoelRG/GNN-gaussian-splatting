#!/usr/bin/env bash
# ============================================================================
# Executa um comando num ambiente sem as variáveis do VSCode snap.
#
# O terminal do VSCode roda dentro de um snap e exporta LOCPATH, GTK_PATH,
# XDG_DATA_DIRS e companhia apontando para /snap/code/.../ e para o core20.
# O COLMAP herda isso, carrega o libpthread do core20 (glibc mais antiga que a
# do sistema) e morre com:
#
#   symbol lookup error: /snap/core20/.../libpthread.so.0:
#   undefined symbol: __libc_pthread_init, version GLIBC_PRIVATE
#
# Num terminal normal o problema não aparece — por isso o convert.py funciona
# quando é você quem roda. Este wrapper reconstrói um ambiente mínimo, mantendo
# o conda gaussian_grouping e o acesso ao display, e chama o comando pedido.
#
# Uso: ./run_colmap_env.sh python convert.py -s data/bonsai
# ============================================================================
USUARIO="$(id -un)"
UID_ATUAL="$(id -u)"
CASA="$(getent passwd "${USUARIO}" | cut -d: -f6)"
CONDA_ENV="${CONDA_ENV:-${CASA}/miniconda3/envs/gaussian_grouping}"

exec env -i \
  PATH="${CONDA_ENV}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  HOME="${CASA}" \
  USER="${USUARIO}" \
  SHELL=/bin/bash \
  LANG=C.UTF-8 \
  DISPLAY="${DISPLAY:-:0}" \
  XAUTHORITY="/run/user/${UID_ATUAL}/gdm/Xauthority" \
  XDG_RUNTIME_DIR="/run/user/${UID_ATUAL}" \
  CONDA_PREFIX="${CONDA_ENV}" \
  CONDA_DEFAULT_ENV=gaussian_grouping \
  "$@"
