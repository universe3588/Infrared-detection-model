%RUN_ALL_FIGURES  一键生成 Ex08 / 消融相关图
% 在 MATLAB 中：
%   cd('.../images/Matlab')
%   run_all_figures

here = fileparts(mfilename('fullpath'));
addpath(here);
cd(here);

fprintf('======== generate paper figures ========\n');
plot_criterion_bar();
plot_distill_bar();
plot_knockout_curves('schematic', true);

fprintf('======== done ========\n');
fprintf('PNG/PDF/FIG -> %s\n', fullfile(here, 'figures'));
