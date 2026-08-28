# Overleaf 与 RASST Git 同步

RASST 的可编辑论文源文件直接由本仓库跟踪在 `paper/` 下，其中主 LaTeX
目录为 `paper/latex/`，图片位于 `paper/latex/figures/`。对应的 Overleaf
项目为 [EMNLP 26 RASST](https://www.overleaf.com/project/6a08a5f98270c576002d785d)，
Git endpoint 为：

```text
https://git@git.overleaf.com/6a08a5f98270c576002d785d
```

当前 Git import 对应 Overleaf `master` 提交
`c96fc4b36dbc22155cdfd832cfa50b548cbfd88b`。`paper/` 使用 Git subtree
接入，因此其中的文件是 RASST 仓库直接跟踪的普通文件，不需要额外初始化
submodule。

## 首次配置

在 Overleaf 的 **Integrations → Git** 中生成 authentication token。Git
询问用户名时使用 `git`，密码使用该 token。token 只应保存在本机 credential
helper 中，不得写进 remote URL、脚本、文档或 commit。

在新的 RASST checkout 中添加 Overleaf remote：

```bash
git remote add overleaf https://git@git.overleaf.com/6a08a5f98270c576002d785d
git fetch overleaf master
```

## 从 Overleaf 拉取

先确保 RASST 工作区干净并位于要更新的分支，然后运行：

```bash
git fetch overleaf master
git subtree pull --prefix=paper overleaf master --squash
```

该命令会把 Overleaf `master` 的新内容合并到 `paper/`，并保留 Overleaf
revision 作为 subtree metadata。检查 diff 和 LaTeX 编译结果后，再将 RASST
分支 push 到 GitHub。

## 从 RASST 推送到 Overleaf

先从 Overleaf 拉取最新内容并解决可能的冲突，再运行：

```bash
git subtree push --prefix=paper overleaf master
```

只应在 `paper/` 内修改论文项目文件；`paper/` 外的 RASST 代码、实验记录和
release 文档不会被推送到 Overleaf。
