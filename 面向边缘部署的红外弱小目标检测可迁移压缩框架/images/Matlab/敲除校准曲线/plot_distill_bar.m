function plot_distill_bar()
%PLOT_DISTILL_BAR  DNA-Net 蒸馏恢复消融（NUAA + NUDT 实测）

here = fileparts(mfilename('fullpath'));
T = readtable(fullfile(here, 'data', 'distill_ablation.csv'));

% 使用：A1/A2/A3 + A4_main（主表）+ Teacher 参考线
idx = ismember(T.setting, {'A1_no_retrain','A2_seg_only','A3_seg_region','A4_full_main'});
S = T(idx, :);
% 保序
order = {'A1_no_retrain','A2_seg_only','A3_seg_region','A4_full_main'};
nuaa = zeros(numel(order),1); nudt = nuaa;
for i = 1:numel(order)
    r = S(strcmp(S.setting, order{i}), :);
    nuaa(i) = r.nuaa_iou; nudt(i) = r.nudt_iou;
end
teacher_nuaa = T.nuaa_iou(strcmp(T.setting,'Teacher'));
teacher_nudt = T.nudt_iou(strcmp(T.setting,'Teacher'));

labels = {'No retrain'; 'Seg only'; 'Seg+Region'; 'Seg+Region+KD'};
Y = [nuaa, nudt];

fig = figure('Color','w','Position',[100 100 780 440]);
b = bar(Y, 0.85);
b(1).FaceColor = [0.20 0.45 0.70];
b(2).FaceColor = [0.85 0.55 0.20];
hold on;
yline(teacher_nuaa, '--', 'Color', [0.20 0.45 0.70], 'LineWidth', 1.2);
yline(teacher_nudt, '-.', 'Color', [0.85 0.55 0.20], 'LineWidth', 1.2);
set(gca, 'XTick', 1:4, 'XTickLabel', labels, ...
    'FontName', 'Times New Roman', 'FontSize', 12, 'LineWidth', 1.0);
ylabel('IoU', 'FontSize', 13);
ylim([0 1.0]);
grid on; box on;
legend({'NUAA-SIRST','NUDT-SIRST', ...
    sprintf('Teacher NUAA=%.4f', teacher_nuaa), ...
    sprintf('Teacher NUDT=%.4f', teacher_nudt)}, ...
    'Location', 'northwest', 'FontSize', 9);
title('Distillation ablations (DNA-Net)', 'FontSize', 13);

outDir = fullfile(here, 'figures');
if ~exist(outDir, 'dir'); mkdir(outDir); end
print(fig, fullfile(outDir, 'fig_distill_bar.png'), '-dpng', '-r300');
print(fig, fullfile(outDir, 'fig_distill_bar.pdf'), '-dpdf', '-bestfit');
savefig(fig, fullfile(outDir, 'fig_distill_bar.fig'));
fprintf('[ok] distill bar -> %s\n', outDir);
end
