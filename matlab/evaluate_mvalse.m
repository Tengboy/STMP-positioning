function evaluate_mvalse(observation_file,output_file,trial_count,worker_count)
% The historical M2VALSE_tby2 kernel, fed the common saved observations.
if nargin<4, worker_count=0; end
root=fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root,'baseline'));
folder=fileparts(output_file); if ~isfolder(folder), mkdir(folder); end
d=load(observation_file);
available=size(d.Y,1);
if nargin<3 || isempty(trial_count), trial_count=available; end
assert(trial_count<=available && trial_count>0);
assert(isequal(size(d.Y),[available,16,64]));
observation_sha256=file_hash(observation_file);
trial_id=d.trial_id(1:trial_count);
channel_index=d.channel_index(1:trial_count);
snr_db=d.snr_db; kappa_db=d.kappa_db;
if isfile(output_file)
    previous=load(output_file);
    assert(strcmp(previous.observation_sha256,observation_sha256),'Observation changed');
    assert(isequal(previous.trial_id,trial_id),'Trial range changed');
    squared_error=previous.squared_error; position=previous.position;
    status=previous.status; direction_flag=previous.direction_flag;
    iterations=previous.iterations; seconds=previous.seconds; error_message=previous.error_message;
else
    squared_error=nan(trial_count,1); position=nan(trial_count,3);
    status=zeros(trial_count,1,'uint8'); direction_flag=false(trial_count,1);
    iterations=zeros(trial_count,1); seconds=nan(trial_count,1); error_message=cell(trial_count,1);
end
if worker_count>0 && isempty(gcp('nocreate'))
    cluster=parcluster('Processes');
    cluster.JobStorageLocation=fullfile(fileparts(output_file),'..','..','cache','baseline_pool');
    if ~isfolder(cluster.JobStorageLocation), mkdir(cluster.JobStorageLocation); end
    parpool(cluster,worker_count);
end
pending=find(status==0); timer=tic;
for start=1:50:numel(pending)
    indices=pending(start:min(start+49,numel(pending)));
    n=numel(indices); chunks=cell(n,1);
    Y=d.Y(indices,:,:); H=d.H_LoS(indices,:,:); P=d.UE_pos(indices,:);
    noise_variance=d.noise_variance;
    parfor (j=1:n,worker_count)
        chunks{j}=run_one(squeeze(Y(j,:,:)),squeeze(H(j,:,:)),P(j,:),noise_variance,kappa_db);
    end
    for j=1:n
        idx=indices(j); item=chunks{j};
        squared_error(idx)=item.error; position(idx,:)=item.position;
        status(idx)=item.status; direction_flag(idx)=item.direction_flag;
        iterations(idx)=item.iterations; seconds(idx)=item.seconds; error_message{idx}=item.message;
    end
    schema_version=1;
    algorithm='M2VALSE_tby2, ha=2, L=16, max_iterations=100, strongest amplitude';
    status_legend='0 pending; 1 finite estimate; 2 no path; 3 nonfinite estimate; 4 exception';
    temporary=[output_file '.partial'];
    save(temporary,'squared_error','position','status','direction_flag','iterations','seconds','error_message',...
        'trial_id','channel_index','snr_db','kappa_db','observation_sha256','algorithm','status_legend','schema_version','-v7');
    movefile(temporary,output_file,'f');
    fprintf('MVALSE SNR=%g completed=%d/%d elapsed_s=%.1f finite_rmse=%.6g flagged_direction=%d\n',...
        snr_db,nnz(status),trial_count,toc(timer),sqrt(mean(squared_error(status==1))),nnz(direction_flag));
end
end

function item=run_one(y,h,truth,noise_variance,kappa_db)
timer=tic;
item=struct('error',inf,'position',[nan nan nan],'status',uint8(4),...
    'direction_flag',false,'iterations',0,'seconds',nan,'message','');
try
    h=sqrt(10^(kappa_db/10))*h;
    out=M2VALSE_tby2(y(:),(0:63)',(0:15)',2,h(:),64,100,noise_variance);
    item.iterations=out.iterations;
    if isempty(out.freqs)
        item.status=uint8(2);
    else
        [~,selected]=max(abs(out.amps));
        mu=out.freqs(selected,:)/pi;
        item.direction_flag=sum(mu.^2)>=1 || mu(2)>=0;
        % Preserve the historical projection and count every finite wrong estimate.
        direction=[mu(1),sqrt(max(0,1-sum(mu.^2))),mu(2)];
        direction=direction/norm(direction);
        item.position=[0,0,25]+(1.5-25)/direction(3)*direction;
        if all(isfinite(item.position))
            item.error=sum((item.position-truth).^2); item.status=uint8(1);
        else
            item.status=uint8(3);
        end
    end
catch exception
    item.message=exception.message;
end
item.seconds=toc(timer);
end

function result=file_hash(path)
md=java.security.MessageDigest.getInstance('SHA-256');
f=fopen(path,'rb'); cleanup=onCleanup(@()fclose(f));
while ~feof(f)
    bytes=fread(f,8*1024*1024,'*uint8'); md.update(bytes);
end
result=lower(reshape(dec2hex(typecast(md.digest(),'uint8'),2).',1,[]));
end
