function results = sar_sstc(y,x,W,info)
% PURPOSE: computes spatial autoregressive model estimates
%           Ynt = p*Wn*Ynt +gamma*Yn,t-1+rho*WnYn,t-1+ X*b+c + e, using sparse matrix algorithms
% ---------------------------------------------------
%  USAGE: results = sar_sstc(y,x,W,info)
%  where:  y = dependent variable vector
%          x = explanatory variables matrix
%          W = standardized contiguity matrix 
%       info = an (optional) structure variable with input options:
%       info.rmin  = (optional) minimum value of rho to use in search (default = -1) 
%       info.rmax  = (optional) maximum value of rho to use in search (default = +1)    
%       info.eig   = 0 for default rmin = -1,rmax = +1, 1 for eigenvalue calculation of these
%       info.convg = (optional) convergence criterion (default = 1e-8)
%       info.maxit = (optional) maximum # of iterations (default = 500)
%       info.lflag = 0 for full lndet computation (default = 1, fastest)
%                  = 1 for MC lndet approximation (fast for very large problems)
%                  = 2 for Spline lndet approximation (medium speed)
%       info.order = order to use with info.lflag = 1 option (default = 50)
%       info.iter  = iterations to use with info.lflag = 1 option (default = 30)  
%       info.lndet = a matrix returned by sar, sar_g, sarp_g, etc.
%                    containing log-determinant information to save time
% ---------------------------------------------------
%  RETURNS: a structure
%         results.meth  = 'sar_sstc'
%         results.theta  = estimates vector before bias correction with the
%         order gammma, rho, b, p, sigma 
%         results.theta1  = estimates vector after bias correction with the order gammma, rho, b, p, sigma 
%         results.tstat = asymp t-stat
%         results.yhat  = yhat         (nobs x 1) vector
%         results.resid = residuals    (nobs x 1) vector
%         results.lik   = log likelihood
%         results.y     = y data vector
%         results.iter  = # of iterations taken
%         results.rmax  = 1/max eigenvalue of W (or rmax if input)
%         results.rmin  = 1/min eigenvalue of W (or rmin if input)
%         results.lflag = lflag from input
%         results.liter = info.iter option from input
%         results.order = info.order option from input
%         results.limit = matrix of [rho lower95,logdet approx, upper95] intervals
%                         for the case of lflag = 1
%         results.time1 = time for log determinant calcluation
%         results.time2 = time for eigenvalue calculation
%         results.time3 = time for hessian or information matrix calculation
%         results.time4 = time for optimization
%         results.time  = total time taken      
%         results.lndet = a matrix containing log-determinant information
%                          (for use in later function calls to save time)
% --------------------------------------------------
%  NOTES: if you use lflag = 1 or 2, info.rmin will be set = -1 
%                                    info.rmax will be set = 1
%         For n < 1000 you should use lflag = 0 to get exact results                                    
% --------------------------------------------------  
% REFERENCES: Anselin (1988), pages 180-182.
% For lndet information see: Ronald Barry and R. Kelley Pace, 
% "A Monte Carlo Estimator of the Log Determinant of Large Sparse Matrices", 
% Linear Algebra and its Applications", Volume 289, Number 1-3, 1999, pp. 41-54.
% and: R. Kelley Pace and Ronald P. Barry 
% "Simulating Mixed Regressive Spatially autoregressive Estimators", 
% Computational Statistics, 1998, Vol. 13, pp. 397-418.
% ---------------------------------------------------

% REFERENCES: Yu, de Jong and Lee (2008), Journal of Econometrics, 146, pages 118-134
% Quasi-Maximum Likelihood Estimators for Spatial Dynamic Panel Data With Fixed Effects When Both n and T Are Large

fields = fieldnames(info);
nf = length(fields);
for i=1:nf,
    if strcmp(fields{i},'tl')       
        tl = info.tl; % tar lag index
    elseif strcmp(fields{i},'stl')
        stl = info.stl; % star lag index
    elseif strcmp(fields{i},'ted')
        ted = info.ted; % time effect indicator,=0 no time effect, =1 use transformed approach,=2 use direct approach, See Lee and Yu (2007)
    end
end


[~, n]=size(W);
[tempsize, ~]=size(y);

t=tempsize/n-2; 

Wnt=kron(speye(t),W);
nt=n*t;

yt=y(2*n+1:2*n+nt);
ytl=y(n+1:n+nt);
ytll=y(1:nt);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
yt=reshape(yt,n,t);
temp=mean(yt')';
temp=temp*ones(1,t);
yt=yt-temp;
yt=reshape(yt,nt,1);

ytl=reshape(ytl,n,t);
temp=mean(ytl')';
temp=temp*ones(1,t);
ytl=ytl-temp;
ytl=reshape(ytl,nt,1);

ytll=reshape(ytll,n,t);
temp=mean(ytll')';
temp=temp*ones(1,t);
ytll=ytll-temp;
ytll=reshape(ytll,nt,1);

xt=x;
if isempty(x) == 0
    %     xt=Q*xt;
    [junk,kx]=size(xt);
    xt=reshape(xt,n,t,kx);
    for i=1:kx
        temp=mean(xt(:,:,i)')';
        temp=temp*ones(1,t);
        xt(:,:,i)=xt(:,:,i)-temp;
    end
    xt=reshape(xt,nt,kx);
else
    xt=[];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Q=kron(eye(t)-1/t*ones(t,1)*ones(1,t),eye(n));
% yt=Q*yt;
% ytl=Q*ytl;
% 
% 
% xt=x;
% if isempty(x) == 0
%     xt=Q*xt;
% else
%     xt=[];
% end
% 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ysl=Wnt*yt;
ystl=Wnt*ytl;
ystll=Wnt*ytll;

if stl + tl == 4
    zt=[ytl ystl ytll ystll xt]; 
elseif stl + tl == 2
    zt=[ytl ystl xt]; %test=[yt ytl ystl xt]
elseif stl + tl == 1
    if stl == 1, zt=[ystl xt]; else zt=[ytl xt]; end
elseif stl + tl == 0
    error('Wrong Info input,Our model has dynamic term anyway');
else
    error('Double-Check stl & tl # in Info structure ');
end

[junk kz]=size(zt);
[junk kx]=size(xt);
results.kz=kz;

time1 = 0; 
time2 = 0;
time3 = 0;
time4 = 0;

timet = clock; % start the clock for overall timing

% if we have no options, invoke defaults
if nargin == 3
    info.lflag = 1;
end;

% parse input options
[rmin,rmax,convg,maxit,detval,ldetflag,eflag,order,miter,options] = sar_parse(info);

% compute eigenvalues or limits
[rmin,rmax,time2] = sar_eigs(eflag,W,rmin,rmax,n);

% do log-det calculations
[detval,time1] = sar_lndet(ldetflag,W,rmin,rmax,detval,order,miter);

t0 = clock;
          Wy = ysl;
          AI = zt'*zt;
          b0 = AI\(zt'*yt);
          bd = AI\(zt'*Wy);
          e0 = yt - zt*b0;
          ed = Wy - zt*bd;
          epe0 = e0'*e0;
          eped = ed'*ed;
          epe0d = ed'*e0;
%[epe0,eped,epe0d,n,t,rmin,rmax]
% step 1) do regressions
% step 2) maximize concentrated likelihood function;
    options = optimset('fminbnd');
    [p,liktmp,exitflag,output] = fminbnd('f_sar_sstc',rmin,rmax,options,detval,epe0,eped,epe0d,n,t);
   
time4 = etime(clock,t0);

if exitflag == 0
fprintf(1,'\n sar: convergence not obtained in %4d iterations \n',output.iterations);
end;
results.iter = output.iterations;

% step 3) find b,sige maximum likelihood estimates
results.beta = b0 - p*bd; 
results.rho = p; 
bhat = results.beta;
results.sige = (1/nt)*(e0-p*ed)'*(e0-p*ed); 
sige = results.sige;

e = (e0 - p*ed);
yhat = (speye(nt) - p*Wnt)\(zt*bhat);
results.yhat = yhat;
resid = yt - yhat;
results.resid = yt - yhat;

parm = [results.beta
        results.rho
        results.sige];

residr=yt-p*Wnt*yt-zt*bhat; %Added J.P.Elhorst, 31-3-2016
results.respaul=residr; %Added J.P.Elhorst, 31-3-2016
    
info_f2=struct('tl',tl,'stl',stl);
    
    
results.lik = f2_sar_sstc(parm,y,x,W,detval,info_f2);


% if n <= 500
t0 = clock;
Sn=eye(n)-p*W;
Sni= inv(Sn); Gn= W*Sni;
pterm = trace(Gn*Gn + Gn*Gn');
Gnt=kron(speye(t),Gn);
SIG = zeros(kz+2,kz+2);               
SIG(1:kz,1:kz) = (1/sige)*(zt'*zt);    % bhat,bhat 
SIG(1:kz,kz+1) = (1/sige)*zt'*Gnt*zt*bhat;% bhat,rho
SIG(kz+1,1:kz) = SIG(1:kz,kz+1)'; 
SIG(kz+1,kz+1) = (1/sige)*bhat'*zt'*Gnt'*Gnt*zt*bhat + t*pterm;% rho,rho
SIG(kz+2,kz+2) = nt/(2*sige*sige);     %sige,sige
SIG(kz+1,kz+2) = t*(1/sige)*trace(Gn);  % rho,sige
SIG(kz+2,kz+1) = SIG(kz+1,kz+2);
SIG=SIG/nt;
%SIGi=inv(SIG);
SIGi=inv(SIG);


u4=(resid.^2)'*(resid.^2)/(n*t);
OMG = zeros(kz+2,kz+2);               % bhat,bhat
OMG(kz+1,kz+1) = diag(Gn)'*diag(Gn)/n;
OMG(kz+2,kz+2) = 1/(4*sige*sige);     %sige,sige
OMG(kz+1,kz+2) = (1/(2*sige))*trace(Gn)/n;  % rho,sige
OMG(kz+2,kz+1) = OMG(kz+1,kz+2);
OMG=(u4/sige^2-3)*OMG;      

theta = [results.beta
        results.rho
        results.sige];

%tmpplus=diag(abs(SIGi));%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%USE THIS ONE UNDER NORMALITY%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tmpplus=diag(abs(SIGi+SIGi*OMG*SIGi));



% bvec = [results.beta
%         results.rho];
tmps = [theta./(sqrt(tmpplus))]*sqrt(n*t);
results.tstat = tmps;
% results.bstd = sqrt(tmp(1:kz,1));
% results.pstd = sqrt(tmp(kz+1,1));
results.std=sqrt(tmpplus)/sqrt(n*t);
time3 = etime(clock,t0);

In=eye(n);

if stl + tl == 4
    An=Sni*((bhat(1,1)+bhat(3,1))*In+(bhat(2,1)+bhat(4,1))*W);
elseif stl + tl == 2
    An=Sni*(bhat(1,1)*In+bhat(2,1)*W);
elseif stl + tl == 1
    if stl == 1, An=Sni*(bhat(1,1)*W); else An=Sni*(bhat(1,1)*In); end
elseif stl + tl == 0
    error('Wrong Info input,Our model has dynamic term anyway');
else
    error('Double-Check stl & tl # in Info structure ');
end

m=0;
[Rn,Dn]=eig(An);
Jn=zeros(n,n);

for i=1:n,
  if Dn(i,i)>1-1/n;
      Jn(i,i)=Dn(i,i);
    m=m+1;
end
end

if m>=1
    Bn=An-Rn*Jn*inv(Rn);
else
    Bn=An;
end

bias1s=zeros(kz+2,1);
bias1u=zeros(kz+2,1);

if stl + tl == 4
    bias1s(1,1)=(1/n)*trace(inv(In-Bn)*Sni);
    bias1s(2,1)=(1/n)*trace(W*inv(In-Bn)*Sni);
    bias1s(3,1)=(1/n)*trace(inv(In-Bn)*Sni);
    bias1s(4,1)=(1/n)*trace(W*inv(In-Bn)*Sni);
    bias1s(kz+1,1)=(1/n)*trace(Gn*inv(In-Bn)*Sni)*bhat(1)+(1/n)*trace(Gn*W*inv(In-Bn)*Sni)*bhat(2)+(1/n)*trace(Gn);
elseif stl + tl == 2
    bias1s(1,1)=(1/n)*trace(inv(In-Bn)*Sni);
    bias1s(2,1)=(1/n)*trace(W*inv(In-Bn)*Sni);
    bias1s(kz+1,1)=(1/n)*trace(Gn*inv(In-Bn)*Sni)*bhat(1)+(1/n)*trace(Gn*W*inv(In-Bn)*Sni)*bhat(2)+(1/n)*trace(Gn);
elseif stl + tl == 1
    if stl == 1, bias1s(1,1)=(1/n)*trace(W*inv(In-Bn)*Sni); bias1s(kz+1,1)=(1/n)*trace(Gn*W*inv(In-Bn)*Sni)*bhat(1)+(1/n)*trace(Gn); 
    else bias1s(1,1)=(1/n)*trace(inv(In-Bn)*Sni);bias1s(kz+1,1)=(1/n)*trace(Gn*inv(In-Bn)*Sni)*bhat(1)+(1/n)*trace(Gn);
    end
elseif stl + tl == 0
    error('Wrong Info input,Our model has dynamic term anyway');
else
    error('Double-Check stl & tl # in Info structure ');
end

bias1s(kz+2,1)=0.5*inv(sige);

bias1utemp=0.5*t*inv(1-p);

if stl + tl == 4
    bias1u(1,1)=bias1utemp;
    bias1u(2,1)=bias1utemp;
    bias1u(3,1)=bias1utemp;
    bias1u(4,1)=bias1utemp;
    bias1u(kz+1,1)=bias1utemp;
elseif stl + tl == 2
    bias1u(1,1)=bias1utemp;
    bias1u(2,1)=bias1utemp;
    bias1u(kz+1,1)=bias1utemp;
elseif stl + tl == 1
    if stl == 1, bias1u(1,1)=bias1utemp; bias1u(kz+1,1)=bias1utemp; 
    else bias1u(1,1)=bias1utemp;bias1u(kz+1,1)=bias1utemp;
    end
elseif stl + tl == 0
    error('Wrong Info input,Our model has dynamic term anyway');
else
    error('Double-Check stl & tl # in Info structure ');
end

if m>=1
    bias1=bias1s+bias1u*(m/n);
else
    bias1=bias1s;
end

bias=SIGi*bias1;

theta1=theta+SIGi*bias1/t;
results.lik1 = f2_sar_sstc(theta1,y,x,W,detval,info_f2); %Added J.P.Elhorst, 31-3-2016

sigetemp=theta1(kz+2,1);
bhattemp=theta1(1:kz,1);
ptemp=theta1(kz+1,1);
Sntemp=eye(n)-ptemp*W;
Snitemp= inv(Sntemp); Gntemp= W*Snitemp;
ptermtemp = trace(Gntemp*Gntemp + Gntemp*Gntemp');
Gnttemp=kron(speye(t),Gntemp);

SIGtemp = zeros(kz+2,kz+2);               % bhat,bhat
SIGtemp(1:kz,1:kz) = (1/sigetemp)*(zt'*zt);     % bhat,rho
SIGtemp(1:kz,kz+1) = (1/sigetemp)*zt'*Gnttemp*zt*bhattemp;
SIGtemp(kz+1,1:kz) = SIGtemp(1:kz,kz+1)'; % rho,rho
SIGtemp(kz+1,kz+1) = (1/sigetemp)*bhattemp'*zt'*Gnttemp'*Gnttemp*zt*bhattemp + t*ptermtemp;
SIGtemp(kz+2,kz+2) = nt/(2*sigetemp*sigetemp);     %sige,sige
SIGtemp(kz+1,kz+2) = t*(1/sigetemp)*trace(Gntemp);  % rho,sige
SIGtemp(kz+2,kz+1) = SIGtemp(kz+1,kz+2);
SIGtemp=SIGtemp/nt;
SIGitemp=inv(SIGtemp);

yhat1 = (speye(nt) - ptemp*Wnt)\(zt*bhattemp);
results.yhat1 = yhat1;
resid1=yt-yhat1;
results.resid1 = yt - yhat1;

% Next block of lines added by J.P.Elhorst, 20-4-2010
residr1=yt-ptemp*Wnt*yt-zt*bhattemp;
results.respaul1=residr1;
yme=yt-mean(yt);
rsqr2=yme'*yme;
rsqr1 = residr'*residr1;
results.rsqr=1.0-rsqr1/rsqr2; %rsquared
res1=yme;
res2=yhat1-mean(yt);
rsq1=res1'*res2;
rsq2=res1'*res1;
rsq3=res2'*res2;
results.corr2=rsq1^2/(rsq2*rsq3); %corr2

u4temp=(resid1.^2)'*(resid1.^2)/(n*t);
OMGtemp = zeros(kz+2,kz+2);               % bhat,bhat
OMGtemp(kz+1,kz+1) = diag(Gntemp)'*diag(Gntemp)/n;
OMGtemp(kz+2,kz+2) = 1/(4*sigetemp*sigetemp);     %sige,sige
OMGtemp(kz+1,kz+2) = (1/(2*sigetemp))*trace(Gntemp)/n;  % rho,sige
OMGtemp(kz+2,kz+1) = OMGtemp(kz+1,kz+2);
OMGtemp=(u4/sigetemp^2-3)*OMGtemp;      

%tmpplus1=diag(abs(SIGitemp));%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%USE THIS ONE UNDER NORMALITY%%%%%%%%%%%%%%%%%%%%%%%%%%%

tmpplus1=diag(abs(SIGitemp+SIGitemp*OMGtemp*SIGitemp));
results.varcov=(SIGitemp+SIGitemp*OMGtemp*SIGitemp)/nt;

bvec1 = theta1;
tmps1 = [bvec1./(sqrt(tmpplus1))]*sqrt(n*t);
results.tstat1 = tmps1;
results.std1=sqrt(tmpplus1)/sqrt(n*t);


results.t=t;
% return stuff
results.meth = 'sar_sstc';
results.yt = yt;
results.zt = zt; %Added J.P.Elhorst, 20-4-2010
results.nobs = n; 
results.nvar = kz;
results.rmax = rmax;      
results.rmin = rmin;
results.lflag = ldetflag;
results.order = order;
results.miter = miter;
% results.rbar = 1 - (rsqr1/rsqr2); % rbar-squared
results.time = etime(clock,timet);
results.time1 = time1;
results.time2 = time2;
results.time3 = time3;
results.time4 = time4;
results.lndet = detval;
results.bias1=bias1;
results.bias=bias;
results.SIG=SIG;
results.SIGi=SIGi;
results.OMG=OMG;
results.SIG1=SIGtemp;
results.SIGi1=SIGitemp;
results.OMG1=OMGtemp;
results.theta=theta;
results.theta1=theta1;

resid = (1-results.theta(1)-results.theta(3))/2;
results.theta(1)= results.theta(1)+resid;
results.theta(3)= results.theta(3)+resid;
resid = (1-sum(results.theta(1:5)))/2;
results.theta(2)= results.theta(2)+resid;
results.theta(4)= results.theta(4)+resid;

function [rmin,rmax,convg,maxit,detval,ldetflag,eflag,order,iter,options] = sar_parse(info)
% PURPOSE: parses input arguments for sar model
% ---------------------------------------------------
%  USAGE: [rmin,rmax,convg,maxit,detval,ldetflag,eflag,order,iter,options] = sar_parse(info)
% where info contains the structure variable with inputs 
% and the outputs are either user-inputs or default values
% ---------------------------------------------------

% set defaults
options = zeros(1,18); % optimization options for fminbnd
options(1) = 0; 
options(2) = 1.e-6; 
options(14) = 500;

eflag = 0;     % default to not computing eigenvalues
ldetflag = 0;  % default to 1999 Pace and Barry MC determinant approx
order = 50;    % there are parameters used by the MC det approx
iter = 30;     % defaults based on Pace and Barry recommendation
rmin = -1;     % use -1,1 rho interval as default
rmax = 1;
detval = 0;    % just a flag
convg = 0.0001;
maxit = 500;

fields = fieldnames(info);
nf = length(fields);
if nf > 0
    
 for i=1:nf
    if strcmp(fields{i},'rmin')
        rmin = info.rmin;  eflag = 0;
    elseif strcmp(fields{i},'rmax')
        rmax = info.rmax; eflag = 0;
    elseif strcmp(fields{i},'convg')
        options(2) = info.convg;
    elseif strcmp(fields{i},'maxit')
        options(14) = info.maxit;  
    elseif strcmp(fields{i},'lndet')
    detval = info.lndet;
    ldetflag = -1;
    eflag = 0;
    rmin = detval(1,1);
    nr = length(detval);
    rmax = detval(nr,1);
    elseif strcmp(fields{i},'lflag')
        tst = info.lflag;
        if tst == 0,
        ldetflag = 0; % compute full lndet, no approximation
        elseif tst == 1,
        ldetflag = 1; % use Pace-Barry approximation
        elseif tst == 2,
        ldetflag = 2; % use spline interpolation approximation
        else
        error('sar: unrecognizable lflag value on input');
        end;
    elseif strcmp(fields{i},'order')
        order = info.order;  
    elseif strcmp(fields{i},'eig')
        eflag = info.eig;  
    elseif strcmp(fields{i},'iter')
        iter = info.iter; 
    end;
 end;
 
else, % the user has input a blank info structure
      % so we use the defaults
end; 

function [rmin,rmax,time2] = sar_eigs(eflag,W,rmin,rmax,n);
% PURPOSE: compute the eigenvalues for the weight matrix
% ---------------------------------------------------
%  USAGE: [rmin,rmax,time2] = far_eigs(eflag,W,rmin,rmax,W)
% where eflag is an input flag, W is the weight matrix
%       rmin,rmax may be used as default outputs
% and the outputs are either user-inputs or default values
% ---------------------------------------------------


if eflag == 1 % do eigenvalue calculations
t0 = clock;
opt.tol = 1e-3; opt.disp = 0;
lambda = eigs(sparse(W),speye(n),1,'SR',opt);  
rmin = real(1/lambda);   
rmax = 1.0;
time2 = etime(clock,t0);
else % use rmin,rmax arguments from input or defaults -1,1
time2 = 0;
end;


function [detval,time1] = sar_lndet(ldetflag,W,rmin,rmax,detval,order,iter);
% PURPOSE: compute the log determinant |I_n - rho*W|
% using the user-selected (or default) method
% ---------------------------------------------------
%  USAGE: detval = far_lndet(lflag,W,rmin,rmax)
% where eflag,rmin,rmax,W contains input flags 
% and the outputs are either user-inputs or default values
% ---------------------------------------------------


% do lndet approximation calculations if needed
if ldetflag == 0 % no approximation
t0 = clock;    
out = lndetfull(W,rmin,rmax);
time1 = etime(clock,t0);
tt=rmin:.001:rmax; % interpolate a finer grid
outi = interp1(out.rho,out.lndet,tt','spline');
detval = [tt' outi];
    
elseif ldetflag == 1 % use Pace and Barry, 1999 MC approximation

t0 = clock;    
out = lndetmc(order,iter,W,rmin,rmax);
time1 = etime(clock,t0);
results.limit = [out.rho out.lo95 out.lndet out.up95];
tt=rmin:.001:rmax; % interpolate a finer grid
outi = interp1(out.rho,out.lndet,tt','spline');
detval = [tt' outi];

elseif ldetflag == 2 % use Pace and Barry, 1998 spline interpolation

t0 = clock;
out = lndetint(W,rmin,rmax);
time1 = etime(clock,t0);
tt=rmin:.001:rmax; % interpolate a finer grid
outi = interp1(out.rho,out.lndet,tt','spline');
detval = [tt' outi];

elseif ldetflag == -1 % the user fed down a detval matrix
    time1 = 0;
        % check to see if this is right
        if detval == 0
            error('sar: wrong lndet input argument');
        end;
        [n1,n2] = size(detval);
        if n2 ~= 2
            error('sar: wrong sized lndet input argument');
        elseif n1 == 1
            error('sar: wrong sized lndet input argument');
        end;          
end;
