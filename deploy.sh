#!/bin/zsh
# One-command GitHub Pages deployment.
# usage: GH_USER=<username> GH_TOKEN=<PAT with repo scope> ./deploy.sh [repo-name]
set -e
GH_USER="${GH_USER:?需要 GH_USER}"
GH_TOKEN="${GH_TOKEN:?需要 GH_TOKEN（classic PAT, repo 权限）}"
REPO="${1:-harness-papers}"
cd "$(dirname "$0")"

echo "── 1/4 创建仓库 $GH_USER/$REPO（如已存在则跳过）"
curl -s -o /dev/null -w "%{http_code}" -X POST \
  -H "Authorization: token $GH_TOKEN" -H "Accept: application/vnd.github+json" \
  https://api.github.com/user/repos -d "{\"name\":\"$REPO\",\"description\":\"2026 具身智能 Harness 范式论文精读库\",\"private\":false}" \
  | grep -q "200\|422" && echo " ok"

echo "── 2/4 推送代码"
git init -q 2>/dev/null || true
git add -A
git commit -qm "Harness Papers: 2026 具身智能论文精读库（25 篇六站深读）" 2>/dev/null || echo " nothing new"
git branch -M main
git remote remove origin 2>/dev/null || true
git remote add origin "https://$GH_USER:$GH_TOKEN@github.com/$GH_USER/$REPO.git"
git push -q -u origin main --force

echo "── 3/4 启用 GitHub Pages"
curl -s -o /dev/null -w "%{http_code}" -X POST \
  -H "Authorization: token $GH_TOKEN" -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/$GH_USER/$REPO/pages" \
  -d '{"source":{"branch":"main","path":"/"}}' | grep -q "200\|201\|409" && echo " ok"

echo "── 4/4 完成 ✅"
echo "站点地址（1-2 分钟内生效）: https://$GH_USER.github.io/$REPO/"
echo "仓库地址: https://github.com/$GH_USER/$REPO"
