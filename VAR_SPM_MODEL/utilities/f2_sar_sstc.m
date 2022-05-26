function llike = f2_sar_sstc(parm,y,x,W,detval,info_f2)
% PURPOSE: evaluates log-likelihood -- given ML estimates
%  spatial autoregressive model using sparse matrix algorithms
% ---------------------------------------------------
%  USAGE:llike = f2_sar(parm,y,X,W,ldet)
%  where: parm = vector of maximum likelihood parameters
%                parm(1:k-2,1) = b, parm(k-1,1) = rho, parm(k,1) = sige
%         y    = dependent variable vector (n x 1)
%         X    = explanatory variables matrix (n x k)
%         W    = spatial weight matrix
%         ldet = matrix with [rho log determinant] values
%                computed in sar.m using one of Kelley Pace's routines  
% ---------------------------------------------------
%  RETURNS: a  scalar equal to minus the log-likelihood
%           function value at the ML parameters
%  --------------------------------------------------
%  NOTE: this is really two functions depending
%        on nargin = 4 or nargin = 5 (see the function)
% ---------------------------------------------------
%  SEE ALSO: sar, f2_far, f2_sac, f2_sem
% ---------------------------------------------------

% written by: James P. LeSage 1/2000
% University of Toledo
% Department of Economics
% Toledo, OH 43606
% jlesage@spatial.econometrics.com


fields = fieldnames(info_f2);
nf = length(fields);
for i=1:nf,
     if strcmp(fields{i},'tl')       
        tl = info_f2.tl; % tar lag index
    elseif strcmp(fields{i},'stl')
        stl = info_f2.stl; % star lag index
    end
end



[junk n]=size(W);
[tempsize junk]=size(y);

t=tempsize/n-2; 

Wnt=kron(speye(t),W);
nt=n*t;

yt=y(2*n+1:2*n+nt);
ytl=y(n+1:n+nt);
ytll=y(1:nt);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Q=kron(speye(t)-1/t*ones(t,1)*ones(1,t),speye(n));
%  
% yt=Q*yt;
% ytl=Q*ytl;
% 
% xt=x;
% 
% xt=x;
% if isempty(x) == 0
%     xt=Q*xt;
% else
%     xt=[];
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


ysl=Wnt*yt;
ystl=Wnt*ytl;
ystll=Wnt*ytll;

if stl + tl == 4
    zt=[ytl ystl ytll ystll xt]; 
elseif stl + tl == 2
    zt=[ytl ystl xt];
elseif stl + tl == 1
    if stl == 1, zt=[ystl xt]; else zt=[ytl xt]; end
elseif stl + tl == 0
    error('Wrong Info input,Our model has dynamic term anyway');
else
    error('Double-Check stl & tl # in Info structure ');
end



[junk kz]=size(zt);
[junk kx]=size(xt);


k = length(parm);
b = parm(1:k-2,1);
rho = parm(k-1,1);
sige = parm(k,1);

gsize = detval(2,1) - detval(1,1);
i1 = find(detval(:,1) <= rho + gsize);
i2 = find(detval(:,1) <= rho - gsize);
i1 = max(i1);
i2 = max(i2);
index = round((i1+i2)/2);
if isempty(index)
index = 1;
end;
detm = detval(index,2);

e = yt-zt*b-rho*ysl;
epe = e'*e;
tmp2 = 1/(2*sige);
llike = -(nt/2)*log(pi) - (nt/2)*log(sige) + t*detm - tmp2*epe;

