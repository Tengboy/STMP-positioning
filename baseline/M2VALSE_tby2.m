function out = M2VALSE_tby2( Y, mx,my, ha, X, Mrx,T,nu)
%MVALSE algorithm for line spectral estimation
% INPUTS:
%   Y  - measurement vector of size M
%   m  - is a vector containing the indices (in ascending order) of the M
%       measurements; subset of {0,1,...,m(end)}
%   ha - indicator determining which approximation of the
%       frequency posterior pdfs will be used:
%           ha=1 will use Heuristic #1
%           ha=2 will use Heuristic #2
%           ha=3 will use point estimation of the frequencies (VALSE-pt)
%   x  - the true signal - used for computing the MSE vs iterations
%   This code is built based on the code written by Mihai Alin Baidu, and is written by QI Zhang and Jiang Zhu.
nu_x = nu;
% nu_y = nu;

[M,G] = size(Y);
Mry   = M/Mrx;
Nx    = mx(Mrx) + 1;    % size of full data in x
Ny    = my(Mry) + 1;    % size of full data in y
N     = Nx * Ny;        % total size
Y_d = reshape(Y,Mry,Mrx,G);
Yx = zeros(Mrx,Mry*G);
for i = 1:G
    Yx(:,(i-1)*Mry+1:i*Mry)=Y_d(:,:,i).';
end

Y2    = Y'*Y;
L     = min(Nx,Ny);          % assumed number of components
% L     = 5;%min(Nx,Ny);
A     = zeros(N,L);
A_x   = zeros(Nx,L);
A_y   = zeros(Ny,L);
J     = zeros(L,L);
J_x   = zeros(L,L);
H     = zeros(L,G);
H_x   = zeros(L,Mry*G);
W     = zeros(L,G);
W_x   = zeros(L,Mry*G);
C     = zeros(L);
C_x   = zeros(L);
mse   = zeros(T,1);
Kt    = zeros(T,1);
t     = 1;

% Initialization of the posterior pdfs of the frequencies
% res   = Y;
res_x = Yx;

for l=1:L
    % noncoherent estimation of the pdf
    %     YI = zeros(N,G);
    YI_x = zeros(Nx,Ny*G);
    YI_y = zeros(Ny,G);
    %     YI(m+1,:) = res;
    YI_x(mx+1,:) = res_x;
    %     YI_y(my+1,:) = res_y;
    R_x = YI_x * YI_x';
    sR_x = zeros(Nx-1,1);
    for i=2:Nx
        for k=1:i-1
            sR_x(i-k) = sR_x(i-k) + R_x(i,k);
        end
    end
    Rh_x  = toeplitz([sum(diag(R_x));sR_x])/Mrx;
    evs_x = sort(real(eig(Rh_x)),'ascend');
    if l==1 % use the sample autocorrelation to initialize the model parameters
        K   = floor(L/2);
        rho = K/L;
        nu_x_0  = mean(evs_x(1:floor(Nx/4)))/G/Mry;
    end
    % nu_x  = mean(evs_x(1:floor(Nx/4)))/G/Mry;
    tau_x = (trace(YI_x'*YI_x)/Mrx/Mry - nu_x*G)/(rho*L);
    % nu_x = nu_x_0;
    etaI_x   = 2*sR_x/(Mrx + nu_x/tau_x)/nu_x;
    ind_x    = find(abs(etaI_x)>0);

    % we omit the point estimation
    [~,mux(l,1),kappax(l,1)] = Heuristic2(etaI_x(ind_x), ind_x);
    A_x(mx+1,l) = exp(1i*mx * mux(l,1)) .* ( besseli(mx,kappax(l,1),1)/besseli(0,kappax(l,1),1) );
    % A_x(mx+1,l) = exp(1i*mx * mux);
    % y方向测量不能联合更新，而应该逐条更新
    % W_temp = W_x(1:l-1,:); C_temp = C_x(1:l-1,1:l-1);
    % J_x(1:l-1,l) = A_x(:,1:l-1)'*A_x(:,l); J_x(l,1:l-1) = J_x(1:l-1,l)'; J_x(l,l) = Mrx;
    % H_x(l,:) = A_x(:,l)'*Yx;
    % v = nu_x / ( Mrx + nu_x/tau_x - real(J_x(1:l-1,l)'*C_temp*J_x(1:l-1,l))/nu_x );
    % u = v .* (H_x(l,:) - J_x(1:l-1,l)'*W_temp)/nu_x;
    % W_x(l,:) = u;
    % ctemp = C_temp*J_x(1:l-1,l)/nu_x;
    % W_x(1:l-1,:) = W_temp - ctemp*u;
    % C_x(1:l-1,1:l-1) = C_temp + v*(ctemp*ctemp');
    % C_x(1:l-1,l) = -v*ctemp;  C_x(l,1:l-1) = C_x(1:l-1,l)'; C_x(l,l) = v;
    
    H_x(l,:) = A_x(:,l)'*YI_x;
    v = nu_x / ( Mrx + nu_x/tau_x);
    u = v .* (H_x(l,:) )/nu_x;
    W_x(l,:) = u;
    % ctemp = C_temp*J_x(1:l-1,l)/nu_x;
    % W_x(1:l-1,:) = W_temp - ctemp*u;
    % C_x(1:l-1,1:l-1) = C_temp + v*(ctemp*ctemp');
    % C_x(1:l-1,l) = -v*ctemp;  C_x(l,1:l-1) = C_x(1:l-1,l)'; C_x(l,l) = v;
    
    %%
    W_x_mat = reshape(W_x(l,:),Mry,G);
    YI_y(my+1,:) = W_x_mat;
    R_y  = YI_y*YI_y';
    sR_y = zeros(Ny-1,1);
    for i=2:Ny
        for k=1:i-1
            sR_y(i-k) = sR_y(i-k) + R_y(i,k);
        end
    end

    Rh_y  = toeplitz([sum(diag(R_y));sR_y])/Mry;
    evs_y = sort(real(eig(Rh_y)),'ascend');
    %         tau_y = tau_x;
    if l==1 % use the sample autocorrelation to initialize the model parameters
        nu_y_0  = mean(evs_y(1:floor(Ny/4)))/G;
        nu = nu_x_0;
    end
    nu_y  = mean(evs_y(1:floor(Ny/4)))/G;
    tau_y = (trace(YI_y'*YI_y)/Mry - nu_y*G)/(rho*L);
    nu_y = nu_y_0;
    if l == 1
        tau = tau_y;
    end
    etaI_y   = 2*sR_y/(Mry + nu_y/tau_y)/nu_y;
    ind_y    = find(abs(etaI_y)>0);
    [~,muy(l,1),kappay(l,1)] = Heuristic2(etaI_y(ind_y), ind_y);
    A_y(my+1,l) = exp(1i*my * muy(l,1)) .* ( besseli(my,kappay(l,1),1)/besseli(0,kappay(l,1),1) );
    % A_y(my+1,l) = exp(1i*my * muy);
    A(:,l) = KR(A_x(:,l),A_y(:,l));


    % use x to initial

    % compute weight estimates; rank one update
    W_temp = W(1:l-1,:); C_temp = C(1:l-1,1:l-1);
    J(1:l-1,l) = A(:,1:l-1)'*A(:,l); J(l,1:l-1) = J(1:l-1,l)'; J(l,l) = M;
    H(l,:) = A(:,l)'*Y;
    v = nu / ( M + nu/tau - real(J(1:l-1,l)'*C_temp*J(1:l-1,l))/nu );
    u = v .* (H(l,:) - J(1:l-1,l)'*W_temp)/nu;
    W(l,:) = u;
    ctemp = C_temp*J(1:l-1,l)/nu;
    W(1:l-1,:) = W_temp - ctemp*u;
    C(1:l-1,1:l-1) = C_temp + v*(ctemp*ctemp');
    C(1:l-1,l) = -v*ctemp;  C(l,1:l-1) = C(1:l-1,l)'; C(l,l) = v;

    % the residual signal
    res = Y - A(:,1:l)*W(1:l,:);
    res_d = reshape(res,Mry,Mrx,G);
    for i = 1:G
        res_x(:,(i-1)*Mry+1:i*Mry)=reshape(res_d(:,:,i).',Mrx,Mry);
    end
    if l==K % save mse and K at initialization
        xro    = A(:,1:l)*W(1:l,:);
        mse(t) = norm(X - xro)^2/norm(X)^2;
        Kt(t)  = K;
    end
end
% nu  = real(trace(Y2 - 2*real(H'*W))/G + trace(W'*J*W)/G + trace(J*C) )/M;
% tau = real(trace(W*W') + G*trace(C))/(K*G);
%%% Start the VALSE algorithm
cont = 1;
while cont
    % for iter_outter = 1:max_outer
    t = t + 1;
    % Update the support and weights
    [ K, s, W, C ] = maxZ( J, H, M, nu, rho, tau );
    % Update the noise variance, the variance of prior and the Bernoulli probability
    if K>0
        % nu  = real(trace(Y2 - 2*real(H(s,:)'*W(s,:)))/G + trace(W(s,:)'*J(s,s)*W(s,:))/G + trace(J(s,s)*C(s,s)) )/M;
        tau = real(trace(W(s,:)*W(s,:)') + G*trace(C(s,s)))/(K*G);
        if K<L
            rho = K/L;
        else
            rho = (L-1)/L; % just to avoid the potential issue of log(1-rho) when rho=1
        end
    else
        rho = 1/L; % just to avoid the potential issue of log(rho) when rho=0
    end

    % Update the posterior pdfs of the frequencies
    inz = 1:L; inz = inz(s); % indices of the non-zero components
    th = zeros(K,2);
    ka = zeros(2,K);
    for i = 1:K
        if K == 1
            R = Y;
            eta = 2/nu * ( R * W(inz,:)' );
            eta_2d = reshape(eta'.',Mry,Mrx);
            eta_x = (eta_2d.' * A_y(:,inz(i)))'.';
            eta_y = (eta_2d*A_x(:,inz(i)))'.';
        else
            A_i = A(:,inz([1:i-1 i+1:end]));
            R = Y - A_i*W(inz([1:i-1 i+1:end]),:);
            eta = 2/nu * ( R * W(inz(i),:)' - A_i * C(inz([1:i-1 i+1:end]),i) );
            eta_2d = reshape(eta'.',Mry,Mrx);
            eta_y = (eta_2d * A_x(:,inz(i)))'.';
            eta_x = (eta_2d.'*A_y(:,inz(i)))'.';
        end
        if ha == 1
            [A_x(:,inz(i)), th(i,1)] = Heuristic1( eta_x, mx, 1000 );
            [A_y(:,inz(i)), th(i,2)] = Heuristic1( eta_y, my, 1000 );
        elseif ha == 2
            [A_x(:,inz(i)), th(i,1), ka(1,i)] = Heuristic2( eta_x, mx );
            [A_y(:,inz(i)), th(i,2), ka(2,i)] = Heuristic2( eta_y, my );
        elseif ha == 3
            [A_x(:,inz(i)), th(i,1)] = pntFreqEst( eta_x, mx );
            [A_y(:,inz(i)), th(i,2)] = pntFreqEst( eta_y, my );
        end
    end
    A = KR(A_x,A_y);
    J(:,s) = A(:,:)'*A(:,s);
    J(s,:) = J(:,s)';
    J(s,s) = J(s,s) - diag(diag(J(s,s))) + M*eye(K);
    H(s,:)   = A(:,s)'*Y;

    % stopping criterion:
    % the relative change of the reconstructed signalis below threshold or
    % max number of iterations is reached
    xr     = A(:,s)*W(s,:);
    mse(t) = norm(xr-X)^2/norm(X)^2;
    Kt(t)  = K;
    if (norm(xr-xro)/norm(xro)<1e-6) || (norm(xro)==0&&norm(xr-xro)==0) || (t >= T)
        cont = 0;
        mse(t+1:end) = mse(t);
        Kt(t+1:end)  = Kt(t);
    end
    xro = xr;
end
out = struct('freqs',th,'kaps',ka,'amps',W(s,:),'x_estimate',xr,'noise_var',nu,'tau',tau,'iterations',t,'mse',mse,'K',Kt);

end

function [a, theta, kappa, mu] = Heuristic1( eta, m, D )
%Heuristic1 Uses the mixture of von Mises approximation of frequency pdfs
%and Heuristic #1 to output a mixture of max D von Mises pdfs

M     = length(m);
tmp   = abs(eta);
A     = besseli(1,tmp,1)./besseli(0,tmp,1);
kmix  = Ainv( A.^(1./m.^2) );
[~,l] = sort(kmix,'descend');
eta_q = 0;
for k=1:M
    if m(l(k)) ~= 0
        if m(l(k)) > 1
            mu2   = ( angle(eta(l(k))) + 2*pi*(1:m(l(k))).' )/m(l(k));
            eta_f = kmix(l(k)) * exp( 1i*mu2 );
        else
            eta_f = eta(l(k));
        end
        eta_q = bsxfun(@plus,eta_q,eta_f.');
        eta_q = eta_q(:);
        kappa = abs(eta_q);

        % to speed up, use the following 4 lines to throw away components
        % that are very small compared to the dominant one
        kmax  = max(kappa);
        ind   = (kappa > (kmax - 30) ); % corresponds to keeping those components with amplitudes divided by the highest amplitude is larger than exp(-30) ~ 1e-13
        eta_q = eta_q(ind);
        kappa = kappa(ind);

        if length(eta_q) > D
            [~, in] = sort(kappa,'descend');
            eta_q   = eta_q(in(1:D));
        end
    end
end
kappa   = abs(eta_q);
mu      = angle(eta_q);
kmax    = max(kappa);
I0reg   = besseli(0,kappa,1) .* exp(kappa-kmax);
Zreg    = sum(I0reg);
n       = 0:1:m(end);
[n1,k1] = meshgrid(n, kappa);
a       = sum( (diag(exp(kappa-kmax))* besseli(n1,k1,1) /Zreg ).*exp(1i*mu*n),1).';
theta   = angle(sum( (diag(exp(kappa-kmax))* besseli(1,kappa,1) /Zreg ).*exp(1i*mu*1),1));
end

function [a, theta, kappa] = Heuristic2( eta, m )
%Heuristic2 Uses the mixture of von Mises approximation of frequency pdfs
%and Heuristic #2 to output one von Mises pdf

N     = length(m);
ka    = abs(eta);
A     = besseli(1,ka,1)./besseli(0,ka,1);
kmix  = Ainv( A.^(1./m.^2) );
k     = N;
eta_q = kmix(k) * exp( 1i * ( angle(eta(k)) + 2*pi*(1:m(k)).' )/m(k) );
for k = N-1:-1:1
    if m(k) ~= 0
        phi   = angle(eta(k));
        eta_q = eta_q + kmix(k) * exp( 1i*( phi + 2*pi*round( (m(k)*angle(eta_q) - phi)/2/pi ) )/m(k) );
    end
end
[~,in] = max(abs(eta_q));
mu     = angle(eta_q(in));
d1     = -imag( eta' * ( m    .* exp(1i*m*mu) ) );
d2     = -real( eta' * ( m.^2 .* exp(1i*m*mu) ) );
if d2<0 % if the function is locally concave (usually the case)
    theta  = mu - d1/d2;
    kappa  = Ainv( exp(0.5/d2) );
else    % if the function is not locally concave (not sure if ever the case)
    theta  = mu;
    kappa  = abs(eta_q(in));
end
n      = (0:1:m(end))';
a      = exp(1i*n * theta).*( besseli(n,kappa,1)/besseli(0,kappa,1) );
end

function [a, theta] = pntFreqEst( eta, m )
%pntFreqEst - point estimation of the frequency

th     = -pi:2*pi/(100*max(m)):pi;

[~,i]  = max(real( eta'*exp(1i*m*th) ));
mu     = th(i);
d1     = -imag( eta' * ( m    .* exp(1i*m*mu) ) );
d2     = -real( eta' * ( m.^2 .* exp(1i*m*mu) ) );
if d2<0 % if the function is locally concave (usually the case)
    theta  = mu - d1/d2;
else    % if the function is not locally concave (not sure if ever the case)
    theta  = mu;
end
a      = exp(1i*(0:1:m(end))' * theta);
end

function [ K, s, W, C ] = maxZ( J, H, M, nu, rho, tau )
%maxZ maximizes the function Z of the binary vector s, see Appendix A of
%the paper

[L,G] = size(H);
cnst = log(rho/(1-rho));

K = 0; % number of components
s = false(L,1); % Initialize s
W = zeros(L,G);
C = zeros(L);
U = zeros(L,G);
v = zeros(L,1);
Delta = zeros(L,1);
if L > 1
    cont = 1;
    while cont
        if K<M-1
            v(~s) = nu ./ ( M + nu/tau - real(sum(J(s,~s).*conj(C(s,s)*J(s,~s)),1))/nu ).';
            %v(~s) = nu ./ ( diag(J(~s,~s)).' + nu/tau - real(sum(J(s,~s).*conj(C(s,s)*J(s,~s)),1))/nu );   %tby
            U(~s,:) = repmat(v(~s),1,G) .* ( H(~s,:) - J(s,~s)'*W(s,:))/nu;
            Delta(~s) = G*log(v(~s)./tau) + diag(U(~s,:)*U(~s,:)')./v(~s) + cnst;
        else
            Delta(~s) = -1; % dummy negative assignment to avoid any activation
        end
        if ~isempty(H(s,:))
            Delta(s) = -G*log(diag(C(s,s))./tau) - diag(W(s,:)*W(s,:)')./diag(C(s,s)) - cnst;
        end
        [~, k] = max(real(Delta));
        if Delta(k)>0
            if s(k)==0 % activate
                W(k,:) = U(k,:);
                ctemp = C(s,s)*J(s,k)/nu;
                W(s,:) = W(s,:) - ctemp*U(k,:);
                C(s,s) = C(s,s) + v(k)*(ctemp*ctemp');
                C(s,k) = -v(k)*ctemp;
                C(k,s) = C(s,k)';
                C(k,k) = v(k);
                s(k) = ~s(k); K = K+1;
            else % deactivate
                s(k) = ~s(k); K = K-1;
                W(s,:) = W(s,:) - C(s,k)*W(k,:)/C(k,k);
                C(s,s) = C(s,s) - C(s,k)*C(k,s)/C(k,k);
            end
            C = (C+C')/2; % ensure the diagonal is real
        else
            break
        end
    end
elseif L == 1
    if s == 0
        v = nu ./ ( M + nu/tau );
        U = v * H/nu;
        Delta = log(v) + u*conj(u)/v + cnst;
        if Delta>0
            W = U; C = v; s = 1; K = 1;
        end
    else
        Delta = -log(C) - W*W'/C - cnst;
        if Delta>0
            W = 0; C = 0; s = 0; K = 0;
        end
    end
end
end

function [ k ] = Ainv( R )
% Returns the approximate solution of the equation R = A(k),
% where A(k) = I_1(k)/I_0(k) is the ration of modified Bessel functions of
% the first kind of first and zero order
% Uses the approximation from
%       Mardia & Jupp - Directional Statistics, Wiley 2000, pp. 85-86.
%
% When input R is a vector, the output is a vector containing the
% corresponding entries

k   = R; % define A with same dimensions
in1 = (R<.53); % indices of the entries < .53
in3 = (R>=.85);% indices of the entries >= .85
in2 = logical(1-in1-in3); % indices of the entries >=.53 and <.85
R1  = R(in1); % entries < .53
R2  = R(in2); % entries >=.53 and <.85
R3  = R(in3); % entries >= .85

% compute for the entries which are < .53
if ~isempty(R1)
    t      = R1.*R1;
    k(in1) = R1 .* ( 2 + t + 5/6*t.*t );
end
% compute for the entries which are >=.53 and <.85
if ~isempty(R2)
    k(in2) = -.4 + 1.39*R2 + 0.43./(1-R2);
end
% compute for the entries which are >= .85
if ~isempty(R3)
    k(in3) = 1./( R3.*(R3-1).*(R3-3) );
end

end