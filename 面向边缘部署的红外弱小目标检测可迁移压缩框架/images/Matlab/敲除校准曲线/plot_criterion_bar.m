function plot_criterion_bar()
%PLOT_CRITERION_BAR  DNA-Net/NUAA 重要性准则对照（实测消融）
% 与 ell_1 / 随机剪枝对照，支撑 I 的有效性解释（Ex08 配套实测图）

here = fileparts(mfilename('fullpath'));
T = readtable(fullfile(here, 'data', 'criterion_nuaa.csv'));

% 投稿主对比：不含 Teacher / 双份 full，full 取主实验
methods = {'Random'; 'L1'; 'I_{task}'; 'I_{contrast}'; 'I (full)'};
ious = [ ...
    T.iou(strcmp(T.method,'Random')); ...
    T.iou(strcmp(T.method,'L1')); ...
    T.iou(strcmp(T.method,'I_task')); ...
    T.iou(strcmp(T.method,'I_contrast')); ...
    T.iou(strcmp(T.method,'I_full_main')) ...
    ];
teacher = T.iou(strcmp(T.method,'Teacher'));

fig = figure('Color','w','Position',[100 100 720 420]);
b = bar(ious, 0.72, 'FaceColor', [0.20 0.45 0.70], 'EdgeColor', 'none');
hold on;
yline(teacher, '--', 'Color', [0.75 0.20 0.20], 'LineWidth', 1.6);
set(gca, 'XTick', 1:numel(methods), 'XTickLabel', methods, ...
    'FontName', 'Times New Roman', 'FontSize', 12, 'LineWidth', 1.0);
ylabel('IoU', 'FontSize', 13);
ylim([0.68 0.78]);
grid on; box on;
legend({'Student (same prune rate + distill)', sprintf('Teacher IoU = %.4f', teacher)}, ...
    'Location', 'northwest', 'FontSize', 10);
title('Importance criterion comparison (DNA-Net / NUAA-SIRST)', 'FontSize', 13);

% 柱顶标注
for i = 1:numel(ious)
    text(i, ious(i) + 0.003, sprintf('%.4f', ious(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 10, 'FontName', 'Times New Roman');
end

outDir = fullfile(here, 'figures');
if ~exist(outDir, 'dir'); mkdir(outDir); end
export_fig_safe(fig, fullfile(outDir, 'fig_criterion_bar'));
fprintf('[ok] criterion bar -> %s\n', outDir);
end

function export_fig_safe(fig, stem)
print(fig, [stem '.png'], '-dpng', '-r300');
print(fig, [stem '.pdf'], '-dpdf', '-bestfit');
savefig(fig, [stem '.fig']);
end
