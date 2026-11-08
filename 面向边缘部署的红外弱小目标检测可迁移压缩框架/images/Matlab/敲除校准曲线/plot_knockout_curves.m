function plot_knockout_curves(varargin)
%PLOT_KNOCKOUT_CURVES  Top-k / Bottom-k IoU 掉点曲线（Ex08）
%
% 用法：
%   plot_knockout_curves
%   plot_knockout_curves('schematic', true)   % 默认 true，图上加水印
%   plot_knockout_curves('schematic', false)  % CSV 已换成实测后关闭水印
%
% CSV 列：remove_ratio, topk_I, bottomk_I, random, l1

p = inputParser;
addParameter(p, 'schematic', true, @islogical);
addParameter(p, 'csv', '', @ischar);
parse(p, varargin{:});

here = fileparts(mfilename('fullpath'));
csvPath = p.Results.csv;
if isempty(csvPath)
    csvPath = fullfile(here, 'data', 'knockout_iou_nuaa.csv');
end
T = readtable(csvPath);
x = T.remove_ratio * 100;

fig = figure('Color', 'w', 'Position', [100 100 760 460]);
hold on;
plot(x, T.topk_I, '-o', 'LineWidth', 1.8, 'MarkerSize', 6, 'Color', [0.75 0.15 0.15]);
plot(x, T.bottomk_I, '-s', 'LineWidth', 1.8, 'MarkerSize', 6, 'Color', [0.15 0.55 0.25]);
plot(x, T.random, '--d', 'LineWidth', 1.5, 'MarkerSize', 5, 'Color', [0.35 0.35 0.35]);
plot(x, T.l1, '-.^', 'LineWidth', 1.5, 'MarkerSize', 5, 'Color', [0.25 0.40 0.75]);

grid on; box on;
xlabel('Removed channels (%)', 'FontName', 'Times New Roman', 'FontSize', 13);
ylabel('IoU', 'FontName', 'Times New Roman', 'FontSize', 13);
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12, 'LineWidth', 1.0);
xlim([0 max(x)]);
ylim([0 0.75]);
legend({'Top-k by I (remove important)', ...
        'Bottom-k by I (remove redundant)', ...
        'Random', 'L1 magnitude'}, ...
        'Location', 'southwest', 'FontSize', 10);
title('Channel knock-out calibration (DNA-Net / NUAA)', 'FontSize', 13);

if p.Results.schematic
    text(0.98, 0.98, ...
        'Units', 'normalized', 'HorizontalAlignment', 'right', ...
        'VerticalAlignment', 'top', 'FontSize', 11, 'FontWeight', 'bold', ...
        'Color', [0.80 0.15 0.15], 'BackgroundColor', [1 1 1]);
end

outDir = fullfile(here, 'figures');
if ~exist(outDir, 'dir'); mkdir(outDir); end
stem = fullfile(outDir, 'fig_knockout_curves');
print(fig, [stem '.png'], '-dpng', '-r300');
print(fig, [stem '.pdf'], '-dpdf', '-bestfit');
savefig(fig, [stem '.fig']);
fprintf('[ok] knockout curves -> %s  (schematic=%d)\n', outDir, p.Results.schematic);
end
