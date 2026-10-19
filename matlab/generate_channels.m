function generate_channels(scenario, split, seed_ids, output_file, worker_count)
% Preserve the historical aggregate generator; add explicit metadata/resume.
base = fileparts(fileparts(mfilename('fullpath')));
if nargin<5, worker_count=0; end
assert(isscalar(worker_count) && worker_count>=0 && worker_count==floor(worker_count));
qsrc = fullfile(base,'third_party','quadriga_src');
addpath(qsrc,'-begin');
addpath(fullfile(qsrc,'equalization_new'),'-begin');
assert(any(strcmp(scenario,{'CDL-C','CDL-A','CDL-B'})));
assert(any(strcmp(split,{'train','val','test'})));
assert(~isempty(seed_ids) && numel(unique(seed_ids))==numel(seed_ids));
seed_ids = reshape(seed_ids,1,[]);
count = numel(seed_ids);
folder=fileparts(output_file); if ~isfolder(folder), mkdir(folder); end
if strcmp(scenario,'CDL-C')
    generator = 'Channel_CDL_TS_specific_v4'; paths = 480;
else
    generator = 'Channel_CDL_TS_specific_v5'; paths = 460;
end
if strcmp(split,'train'), suffix='train'; else, suffix='test'; end
hkey = ['h_f_' suffix];
xkey = ['angle_X_' suffix]; ykey = ['angle_Y_' suffix];
pkey = ['path_power_' suffix];
if exist(output_file,'file')
    old = load(output_file,'scenario','split','seed_ids','completed');
    assert(strcmp(old.scenario,scenario) && strcmp(old.split,split));
    assert(isequal(old.seed_ids,seed_ids),'Resume seeds differ');
    completed = old.completed;
else
    completed = 0;
    frequency_hz = 28e9; dimensions = [count,2,16,64];
    normalization = 'mean(abs(complex_channel).^2)=1 per sample';
    generator_path = which(generator);
    save(output_file,'scenario','split','seed_ids','completed',...
        'frequency_hz','dimensions','normalization','generator','generator_path','-v7.3');
end
m = matfile(output_file,'Writable',true);
if completed==0
    m.(hkey)(count,2,16,64)=0;
    m.(xkey)(count,paths)=0; m.(ykey)(count,paths)=0; m.(pkey)(count,paths)=0;
end
fprintf('START %s %s completed=%d/%d generator=%s\n',scenario,split,completed,count,which(generator));
if worker_count>0 && completed<count
    pool = gcp('nocreate');
    if isempty(pool)
        cluster = parcluster('Processes');
        storage = fullfile(fileparts(output_file),'..','cache','matlab_pool');
        if ~exist(storage,'dir'), mkdir(storage); end
        cluster.JobStorageLocation = storage;
        pool = parpool(cluster,worker_count);
    end
    assert(pool.NumWorkers>=worker_count);
end
timer = tic; initial_completed = completed;
block_size = 250;
for start_idx = completed+1:block_size:count
    stop_idx = min(start_idx+block_size-1,count);
    n = stop_idx-start_idx+1;
    channels = zeros(n,2,16,64);
    angles_x = zeros(n,paths); angles_y = zeros(n,paths); powers = zeros(n,paths);
    parfor (local_idx = 1:n, worker_count)
        idx = start_idx+local_idx-1;
        rng(seed_ids(idx),'twister');
        [h,prior] = feval(generator,['NR-' scenario],1,28e9,zeros(2,1),...
            300*ones(2,1),16,64,1,0.0005,10);
        h = reshape(h,16,64);
        assert(all(isfinite(h(:))) && abs(mean(abs(h).^2,'all')-1)<1e-10);
        assert(numel(prior.rho)==paths && all(isfinite(prior.rho(:))));
        ri = cat(3,real(h),imag(h));
        channels(local_idx,:,:,:) = permute(ri,[4,3,1,2]);
        angles_x(local_idx,:) = reshape(prior.angle_X,1,[]);
        angles_y(local_idx,:) = reshape(prior.angle_Y,1,[]);
        powers(local_idx,:) = reshape(prior.rho,1,[]);
    end
    rows = start_idx:stop_idx;
    m.(hkey)(rows,:,:,:) = channels;
    m.(xkey)(rows,:) = angles_x; m.(ykey)(rows,:) = angles_y; m.(pkey)(rows,:) = powers;
    m.completed = stop_idx;
    fprintf('PROGRESS %s %s %d/%d elapsed_s=%.1f samples_per_s=%.3f\n',...
        scenario,split,stop_idx,count,toc(timer),(stop_idx-initial_completed)/max(toc(timer),eps));
end
fprintf('DONE %s %s count=%d elapsed_s=%.3f\n',scenario,split,count,toc(timer));
end
