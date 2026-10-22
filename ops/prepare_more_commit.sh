#!/usr/bin/env bash
# Default: read-only preflight; --apply-ignore: append bounded ignore rules;
# --stage: stage explicit allowlist only, without committing anything.
set -euo pipefail

DOC_FIX='docs/fixes/2026-10-21-MoRe-inference-fix-log.md'
DOC_ENV='docs/setup/MoRe-CUDA128-environment.md'
FRAGMENT_PATH="${MORE_IGNORE_FRAGMENT:-ops/MoRe.gitignore.fragment}"
COMMIT_HASH='2f23ca2ea9638cecad7f7220795210fc6b8353c3'

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

repo_root=$(git rev-parse --show-toplevel 2>/dev/null) || fail '当前目录不是 Git 仓库'
[[ "$(pwd -P)" == "$(cd "$repo_root" && pwd -P)" ]] || fail "请切换到 Git 根目录：$repo_root"
[[ -f inference.py && -f vggt/layers/attention.py && -f vggt/models/aggregator.py ]] || fail '当前目录不是预期的 MoRe 仓库根目录'
[[ -f "$DOC_FIX" && -f "$DOC_ENV" ]] || fail '请先把两份 docs Markdown 放入相应目录'

case "${1:---check}" in
  --apply-ignore)
    [[ -f "$FRAGMENT_PATH" ]] || fail "找不到 .gitignore 片段：$FRAGMENT_PATH（可用 MORE_IGNORE_FRAGMENT= 指定）"
    [[ ! -e .gitignore || -f .gitignore ]] || fail '.gitignore 不是普通文件，拒绝修改'
    if [[ -f .gitignore ]] && grep -Fq '# BEGIN MORE LOCAL ARTIFACTS (2026-10-21)' .gitignore; then
      echo '.gitignore 已经含有本次规则区块，不重复追加。'
    else
      if [[ -s .gitignore ]]; then printf '\n' >> .gitignore; fi
      cat "$FRAGMENT_PATH" >> .gitignore
      echo '已向仓库 .gitignore 追加带边界标记的规则；请人工审查 git diff -- .gitignore'
    fi
    git status --short
    ;;
  --check|--stage)
    [[ -f .gitignore ]] || fail '缺少仓库 .gitignore；请先使用 --apply-ignore'
    grep -Fq '# BEGIN MORE LOCAL ARTIFACTS (2026-10-21)' .gitignore || fail '.gitignore 缺少本次规则标记'
    for f in requirements-cu128.txt requirements-cu128-constraints.txt requirements-git.txt; do
      [[ -s "$f" ]] || fail "依赖文件缺失/为空：$f"
    done
    grep -Fxq "git+https://github.com/jytime/LightGlue.git@${COMMIT_HASH}#egg=lightglue" requirements-git.txt || fail 'LightGlue 依赖必须锁定这次实测的 commit SHA，并复核文件内容'
    git ls-files --error-unmatch -- requirements.txt >/dev/null 2>&1 || fail '找不到已跟踪的官方 requirements.txt'
    git diff --quiet -- requirements.txt || fail '官方 requirements.txt 有未提交修改，请先单独核查'
    git diff --cached --quiet -- requirements.txt || fail '官方 requirements.txt 在暂存区发生变化，请先处理'
    python - <<'PY'
from pathlib import Path
import re
p = Path('requirements-cu128-constraints.txt')
lines = [x.strip() for x in p.read_text(encoding='utf-8').splitlines() if x.strip() and not x.lstrip().startswith('#')]
if not lines:
    raise SystemExit('依赖约束文件为空')
for line in lines:
    if not re.match(r'^[A-Za-z0-9_.-]+==[A-Za-z0-9_.+!\-]+$', line):
        raise SystemExit('约束文件含非普通版本锁定项，先检查是否有本地路径/URL/凭据：' + line[:120])
    if not re.match(r'^(torch|torchvision|triton|nvidia-.*-cu12)==', line, flags=re.I):
        raise SystemExit('约束文件包含意料之外的包，请人工核对：' + line)
print('约束快照格式检查通过；这不替代对版本与安全性的人工审查。')
PY
    python -m py_compile inference.py vggt/layers/attention.py vggt/models/aggregator.py
    git diff --check
    echo '--- 可提交文件清单 ---'
    printf '%s\n' inference.py vggt/layers/attention.py vggt/models/aggregator.py \
      "$DOC_FIX" "$DOC_ENV" requirements-cu128.txt requirements-cu128-constraints.txt requirements-git.txt .gitignore
    echo '--- 当前状态 ---'
    git status --short
    if [[ "${1:---check}" == '--stage' ]]; then
      git diff --cached --quiet || fail '暂存区已经有变更。先执行 git diff --cached 审查并单独处理，避免合并进本次提交'
      git add -- \
        inference.py \
        vggt/layers/attention.py \
        vggt/models/aggregator.py \
        "$DOC_FIX" \
        "$DOC_ENV" \
        requirements-cu128.txt \
        requirements-cu128-constraints.txt \
        requirements-git.txt \
        .gitignore
      git diff --cached --check
      echo '--- 暂存区统计 ---'
      git diff --cached --stat
      echo '--- 暂存的文件 ---'
      git diff --cached --name-only
      echo '--- 完成暂存：没有执行 git commit。请人工检查 git diff --cached。 ---'
    fi
    ;;
  *)
    fail '用法：bash prepare_more_commit.sh [--check|--apply-ignore|--stage]'
    ;;
esac
