# -*- coding: utf-8 -*-
from pathlib import Path

p = Path(r"d:\Desktop\红外检测模型\面向边缘部署的红外弱小目标检测可迁移压缩框架\bare_jrnl_new_sample4.tex")
text = p.read_text(encoding="utf-8")

marker_start = "\\label{sec:ablation}\n\n"
marker_end = "\\subsection{定性结果}"
i0 = text.find(marker_start)
i1 = text.find(marker_end)
if i0 < 0 or i1 < 0 or i1 <= i0:
    raise SystemExit(f"markers not found: {i0=} {i1=}")

new_body = r"""\label{sec:ablation}

在 DNA-Net 上于 NUAA-SIRST 做准则对照与蒸馏配方消融，并在 NUDT-SIRST 复现蒸馏组。除特别声明外，压缩率与主实验一致。

表~\ref{tab:distill}表明：（1）剪后不重训几乎不可用（NUAA IoU $\approx 0.008$）；（2）仅用 GT SoftIoU 重训已显著超过教师（NUAA $0.7454$ vs $0.7171$；NUDT $0.9149$ vs $0.8487$），说明\textbf{结构化变窄本身提供主要泛化收益}；（3）加入 $\mathcal{L}_{\mathrm{region}}$ 后进一步提升（NUAA $0.7551$，NUDT $0.9180$）；（4）再叠加强输出 KD（$\lambda_{\mathrm{kd}}=1$）时，NUAA 上相对“分割+区域”反而回落至主表完整配方 $0.7387$（仍高于教师 $0.7171$），NUDT 上 $\lambda_{\mathrm{kd}}=1$ 曾崩溃（IoU$\to 0$），降至 $0.5$ 后恢复至 $0.9184$（相对无 KD 仅 $+0.0004$）。

\textbf{教师上限约束（Teacher Ceiling Effect）。}
DNA-Net 教师在 NUAA 上仅 $0.7171$，而学生经结构正则与 $\mathcal{L}_{\mathrm{region}}$ 已达 $0.7551$。此时过大的 $\lambda_{\mathrm{kd}}$ 会把学生输出强行拉向较弱教师分布，形成上限拖累；这解释了表~\ref{tab:distill}中“加 KD 未必再涨”的现象，也说明 DNA 涨点应主要归因于\textbf{结构正则 + 区域损失}。实践上：弱教师/学生已超教师时取较小 $\lambda_{\mathrm{kd}}$（如 NUDT 的 $0.5$）；在更强教师骨干（UIU-Net/SCTransNet）上，输出 KD 主要起稳定与校准作用，故仍保留在式~\eqref{eq:loss}中作为可调项，而非强制等权硬蒸馏。

\begin{table}[!t]
\renewcommand{\arraystretch}{1.12}
\caption{蒸馏恢复消融（DNA-Net；同 keep\_masks）。NUAA 完整配方与主表对齐；NUDT 完整 KD 取 $\lambda_{\mathrm{kd}}=0.5$}
\label{tab:distill}
\centering
\begin{tabular}{@{}lcc@{}}
\toprule
设置 & NUAA IoU & NUDT IoU \\
\midrule
剪后不重训 & 0.0081 & 0.2183 \\
仅 GT 分割损失 & 0.7454 & 0.9149 \\
分割 + 区域损失 & 0.7551 & 0.9180 \\
分割 + 区域 + 输出 KD & 0.7387 & 0.9184 \\
\bottomrule
\end{tabular}
\end{table}

\textbf{评分证据以敲除为主、长蒸馏为辅。}
长蒸馏（$\approx 200$ epoch）会抹平不同剪枝起点的差距，甚至在单次种子下使修调后的 $\ell_1$ 反超（表~\ref{tab:criterion}）。因此评判 $I$ 的主战场是图~\ref{fig:knockout}的不重训敲除曲线：按本文 $I$ 去掉 Top-$k$ 通道时 IoU 急剧下降，去掉 Bottom-$k$ 时近乎平坦，随机与 $\ell_1$ 介于其间——说明 $I$ 能正确区分重要/冗余通道。表~\ref{tab:criterion}在同压缩率+完整蒸馏下仅作辅助对照：随机约 $0.72$；仅 $I_{\mathrm{task}}$/$I_{\mathrm{contrast}}$ 为 $0.7374$/$0.7408$；完整 $I$ 与主表对齐为 $0.7387$。$\ell_1$ 在默认同超参下首次崩溃（IoU$\to 0$），经额外调参挽救后才达 $0.7595^{\ast}$，说明其破坏浅层关键通路、初始点病态；本文评分无需针对特定骨干额外调参即可稳定收敛。跨骨干一致性与板端收益仍是协议主体证据。
全文默认 $\alpha=0.7$，分层配额采用浅层保护、深层多剪的经验设定（DNA-Net 为 $p=(0.2,0.3,0.4,0.5,0.6)$）；$\alpha=0/1$ 的极端情形已由仅 $I_{\mathrm{contrast}}$ / 仅 $I_{\mathrm{task}}$ 消融覆盖，故不再单独绘制 $\alpha$ 与 $\{p_s\}$ 敏感性曲线。

\begin{table}[!t]
\renewcommand{\arraystretch}{1.12}
\caption{重要性准则对照（相同压缩率 + 完整蒸馏，DNA-Net/NUAA；评分主证据见图~\ref{fig:knockout}）}
\label{tab:criterion}
\centering
\begin{tabular}{@{}lc@{}}
\toprule
准则 & $\mathrm{IoU}\uparrow$ \\
\midrule
随机剪枝 & 0.7247 \\
$\ell_1$ 权重（默认同超参） & $0^{\dagger}$ \\
$\ell_1$ 权重（调参挽救） & 0.7595$^{\ast}$ \\
仅 $I_{\mathrm{task}}$ & 0.7374 \\
仅 $I_{\mathrm{contrast}}$ & 0.7408 \\
本文 $I$（完整，与主表对齐） & 0.7387 \\
\bottomrule
\end{tabular}\\[0.3em]
{\footnotesize $^{\dagger}$首次训练 IoU$\to 0$ 崩溃；$^{\ast}$额外调参后重跑，不作为默认可复现基线。}
\end{table}

\begin{figure}[!t]
\centering
\includegraphics[width=0.92\columnwidth]{fig_criterion_bar.png}
\caption{DNA-Net/NUAA 重要性准则对照（同压缩率+完整蒸馏；虚线为教师 IoU；$\ell_1$* 为调参挽救结果）。长蒸馏下数值接近，评分因果证据见图~\ref{fig:knockout}。}
\label{fig:criterion}
\end{figure}

\begin{figure}[!t]
\centering
\includegraphics[width=0.92\columnwidth]{fig_distill_bar.png}
\caption{DNA-Net 蒸馏恢复消融（NUAA/NUDT）。NUAA 上“分割+区域”已超教师；再加强 KD 受教师上限约束。}
\label{fig:distill}
\end{figure}

\begin{figure}[!t]
\centering
\includegraphics[width=0.92\columnwidth]{fig_knockout_curves.png}
\caption{\textbf{评分主证据}：通道敲除校准曲线（DNA-Net/NUAA 教师，不重训）。按重要性 $I$ 去掉 Top-$k$ 通道时 IoU 下降最快；去掉 Bottom-$k$ 冗余通道时下降最慢；随机与 $\ell_1$ 介于二者之间。}
\label{fig:knockout}
\end{figure}

"""

text2 = text[:i0] + new_body + text[i1:]
p.write_text(text2, encoding="utf-8")
print("patched ablation section ok; new len", len(text2))
