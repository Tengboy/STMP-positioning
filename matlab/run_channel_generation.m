function run_channel_generation(run_dir,workers,mode)
% Generate the disjoint source splits and frozen transfer test scenarios.
base=fileparts(fileparts(mfilename('fullpath')));
if nargin<1, run_dir=fullfile(base,'work'); end
if nargin<2, workers=8; end
if nargin<3, mode='full'; end
assert(any(strcmp(mode,{'full','smoke'})));
if strcmp(mode,'full'), ntest=3000; nval=2000; ntrain=50000;
else, ntest=2; nval=2; ntrain=4; end
addpath(fullfile(base,'matlab'));
generate_channels('CDL-C','test',1:ntest,fullfile(run_dir,'datasets','CDLC_test.mat'),workers);
generate_channels('CDL-C','val',3001:3000+nval,fullfile(run_dir,'datasets','CDLC_val.mat'),workers);
generate_channels('CDL-C','train',5001:5000+ntrain,fullfile(run_dir,'datasets','CDLC_train.mat'),workers);
generate_channels('CDL-A','test',1:ntest,fullfile(run_dir,'datasets','CDLA_test.mat'),workers);
generate_channels('CDL-B','test',1:ntest,fullfile(run_dir,'datasets','CDLB_test.mat'),workers);
fprintf('ALL_CHANNEL_DATASETS_COMPLETE mode=%s\n',mode);
end
