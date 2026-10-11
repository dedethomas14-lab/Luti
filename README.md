# LUTI Classroom Demo：最小教学增量版

本目录以原始 Streamlit `app.py` 为基础，只增加计算过程教学功能。`app_original.py` 是未修改备份。

## 启动

Windows PowerShell 5.1：

```powershell
Set-Location "C:\Users\Lenovo\Documents\Codex\2026-10-11\files-mentioned-by-the-user-codex\outputs\luti-streamlit-teaching-v2"
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
& .\.venv\Scripts\python.exe -m streamlit run .\app.py
```

## 测试

```powershell
python -m unittest discover -s tests -v
python -m py_compile app.py teaching_math.py
```

## 新增交互

- 展开 `Teaching Mode / Step-by-Step Calculation`，选择起点、终点和要编辑的小区。
- 使用原生输入框只修改一个小区的人口和就业，原三张 Plotly 图同步更新。
- 不同区域的编辑按 Zone 保存在会话中；切换区域、方式、β 或约束不会丢失。`Restore initial city data` 可一键恢复。
- 用步骤选择器逐项展示当前模式的出行生成、吸引力、衰减、权重、概率与 OD 流量。
- 默认展示确定性期望流量；侧栏可显式启用固定种子的 Poisson 模拟。
- 在 `Doubly-Constrained` 下勾选 `Show balancing process`，逐帧查看 IPF 行列平衡。
- 双约束默认采用 `Teaching Scenario`：`Dⱼ = ΣO × Jobsⱼ / ΣJobs`；可切回 `Original` 保留原列和目标。
- 播放时 Flow Map、公式和所选 OD 指标共用当前帧；金色高亮不受流量百分位阈值影响。
- 可直接跳到初始、最终或任意帧；完整 40×40 矩阵收在详细展开区。
- `Fast classroom demo` 只播放真实历史中的首轮行约束、第 1、2、5、10、20、50 轮及最终帧。
- 初始矩阵明确标为无“次出行”单位的权重；首次行约束后才使用 trips（次出行）。
- IPF 失败时明确回退到起点约束矩阵，不把全零矩阵伪装成有效结果。

数学修改和舍弃范围见 `SCOPE_ADJUSTMENT.md`。
