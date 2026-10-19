function run_baseline(run_dir,workers,count)
% Non-official MDVALSE baseline on exactly the shared Python observations.
base=fileparts(fileparts(mfilename('fullpath')));
if nargin<1, run_dir=fullfile(base,'work'); end
if nargin<2, workers=8; end
if nargin<3, count=3000; end
addpath(fullfile(base,'matlab'));
for scenario={'CDLC','CDLB','CDLA'}
    for snr=[0 4 8 12 16 20]
        name=sprintf('%s_k0_s%d',scenario{1},snr);
        evaluate_mvalse(fullfile(run_dir,'observations',[name '.mat']),...
            fullfile(run_dir,'results','raw',['MVALSE_' name '.mat']),count,workers);
    end
end
fprintf('ALL_MDVALSE_EXPERIMENTS_COMPLETE\n');
end
