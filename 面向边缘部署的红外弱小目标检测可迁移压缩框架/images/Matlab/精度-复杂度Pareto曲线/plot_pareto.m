function plot_pareto()
%PLOT_PARETO  Fig.2 IoU--Params / IoU--FLOPs（教师→学生连线）
% 数据：data/pareto_teacher_student.csv（主表 IoU 均值 + 资源表复杂度）

here = fileparts(mfilename('fullpath'));
T = readtable(fullfile(here, 'data', 'pareto_teacher_student.csv'));

backs = unique(T.backbone, 'stable');
% 固定顺序
order = {'DNA-Net','UIU-Net','MSHNet','SCTransNet'};
backs = order(ismember(order, backs));

colors = [ ...
    0.75 0.20 0.20; ...  % DNA
    0.20 0.45 0.75; ...  % UIU
    0.15 0.55 0.30; ...  % MSH
    0.55 0.30 0.65];     % SCTrans

fig = figure('Color','w','Position',[80 80 980 420]);

% ---- left: IoU vs Params ----
subplot(1,2,1); hold on;
for i = 1:numel(backs)
    bb = backs{i};
    t = T(strcmp(T.backbone, bb) & strcmp(T.role, 'teacher'), :);
    s = T(strcmp(T.backbone, bb) & strcmp(T.role, 'student'), :);
    plot([t.params_M, s.params_M], [t.iou_mean, s.iou_mean], '-', ...
        'Color', colors(i,:), 'LineWidth', 1.8);
    % arrow teacher -> student
    quiver(t.params_M, t.iou_mean, s.params_M - t.params_M, s.iou_mean - t.iou_mean, 0, ...
        'Color', colors(i,:), 'LineWidth', 1.2, 'MaxHeadSize', 0.35, 'HandleVisibility', 'off');
    plot(t.params_M, t.iou_mean, 'o', 'MarkerSize', 9, 'MarkerFaceColor', colors(i,:), ...
        'MarkerEdgeColor', 'k', 'LineWidth', 0.8);
    plot(s.params_M, s.iou_mean, 's', 'MarkerSize', 9, 'MarkerFaceColor', 'w', ...
        'MarkerEdgeColor', colors(i,:), 'LineWidth', 1.8);
end
grid on; box on;
set(gca, 'XScale', 'log', 'FontName', 'Times New Roman', 'FontSize', 11, 'LineWidth', 1.0);
xlabel('Params (M)', 'FontSize', 12);
ylabel('Mean IoU', 'FontSize', 12);
title('(a) IoU--Params', 'FontSize', 12);
xlim([0.6 80]);
ylim([0.70 0.81]);

% ---- right: IoU vs FLOPs ----
subplot(1,2,2); hold on;
hTeach = gobjects(0); hStud = gobjects(0); hLine = gobjects(numel(backs),1);
for i = 1:numel(backs)
    bb = backs{i};
    t = T(strcmp(T.backbone, bb) & strcmp(T.role, 'teacher'), :);
    s = T(strcmp(T.backbone, bb) & strcmp(T.role, 'student'), :);
    hLine(i) = plot([t.flops_G, s.flops_G], [t.iou_mean, s.iou_mean], '-', ...
        'Color', colors(i,:), 'LineWidth', 1.8, 'DisplayName', bb);
    quiver(t.flops_G, t.iou_mean, s.flops_G - t.flops_G, s.iou_mean - t.iou_mean, 0, ...
        'Color', colors(i,:), 'LineWidth', 1.2, 'MaxHeadSize', 0.35, 'HandleVisibility', 'off');
    plot(t.flops_G, t.iou_mean, 'o', 'MarkerSize', 9, 'MarkerFaceColor', colors(i,:), ...
        'MarkerEdgeColor', 'k', 'LineWidth', 0.8, 'HandleVisibility', 'off');
    plot(s.flops_G, s.iou_mean, 's', 'MarkerSize', 9, 'MarkerFaceColor', 'w', ...
        'MarkerEdgeColor', colors(i,:), 'LineWidth', 1.8, 'HandleVisibility', 'off');
end
% 图例用色块代理 Teacher/Student 标记
hTeach = plot(nan, nan, 'o', 'MarkerSize', 8, 'MarkerFaceColor', [0.4 0.4 0.4], ...
    'MarkerEdgeColor', 'k', 'DisplayName', 'Teacher');
hStud = plot(nan, nan, 's', 'MarkerSize', 8, 'MarkerFaceColor', 'w', ...
    'MarkerEdgeColor', [0.2 0.2 0.2], 'LineWidth', 1.6, 'DisplayName', 'Student');
grid on; box on;
set(gca, 'XScale', 'log', 'FontName', 'Times New Roman', 'FontSize', 11, 'LineWidth', 1.0);
xlabel('FLOPs (G)', 'FontSize', 12);
ylabel('Mean IoU', 'FontSize', 12);
title('(b) IoU--FLOPs', 'FontSize', 12);
xlim([3 150]);
ylim([0.70 0.81]);
legend([hLine; hTeach; hStud], 'Location', 'southwest', 'FontSize', 9);

% 总标题
sgtitle('Accuracy--Complexity Pareto (Teacher \rightarrow Student)', ...
    'FontName', 'Times New Roman', 'FontSize', 13, 'FontWeight', 'bold');

outDir = fullfile(here, 'figures');
if ~exist(outDir, 'dir'); mkdir(outDir); end
stem = fullfile(outDir, 'fig_pareto');
print(fig, [stem '.png'], '-dpng', '-r300');
print(fig, [stem '.pdf'], '-dpdf', '-bestfit');
savefig(fig, [stem '.fig']);
fprintf('[ok] Pareto -> %s\n', outDir);
end
