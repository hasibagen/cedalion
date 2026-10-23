# jade fork 说明

这是 ibs-lab/cedalion 的个人 fork，归 hasibagen 维护。用途、分支模型和纪律：

- 用途：承载厂商 reader。第一个是 `src/cedalion/io/yiruide.py`（依瑞德 zlhk
  CSV），后续加日立（rili）等 CSV 类格式。规划文档在
  `/home/jade/toolbox/fnirs/cedalion_ext/docs/plans/cedalion-toolset/`
  （decisions/0002）。
- 分支模型：`jade-io` 基于 `v26.5.1`，是日常安装来源；上游合并时
  `git fetch upstream && git merge upstream/main`。`main` 分支保持与上游同步，
  不放个人改动。
- 最小改动纪律：只新增文件（io/yiruide.py、tests/test_io_yiruide.py、本文件）；
  `io/__init__.py` 只加导入行；不改上游任何其他代码。个人分析代码放
  cedalion_ext 包，不进本 fork。
- 发布：改动后打 tag（如 `v26.5.1.post1`），构建 wheel 上传到 GitHub Release，
  各台机器 pip 装 Release 里的 wheel；整仓 git clone 有 483 MB，慢网络下不作为
  安装方式。
- 提 PR 给官方时：从 jade-io 里抽出 io/ 相关 commit，cherry-pick 到基于上游
  main 的干净分支再提。
