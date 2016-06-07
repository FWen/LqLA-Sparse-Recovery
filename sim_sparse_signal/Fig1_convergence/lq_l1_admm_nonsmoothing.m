function [x,Out] = lq_l1_admm_nonsmoothing(A,y,lamda,q,xtrue,rho,x0,max_iter);
% lq_l1_admm_nonsmoothing solves
%
%   minimize || Ax - y ||_1 + \lamda || x ||_q^q
%
% Inputs:
%	A: sensing matrix
%	y: CS data
%	0<=q<=1
%	lamda: regularization parameter 
%	x0: initialization 
%	xtrue: for debug, for calculation of errors
% Outputs
%	x: the CS recovery
%	Out.e: the error with respect to the true
%	Out.et: time index


%Convergence setup
if nargin<8
    max_iter = 10000;
end

ABSTOL = 1e-7;

if(isobject(A))
    m = A.m;
    n = A.n;
else
    [m,n]=size(A);
    At = A';
    A2 = A'*A;
end

if nargin<6
	rho = 1e3;
end
    
%Initialize
if nargin<7
	x  = zeros(n,1);
else
	x = x0;
end;

w = zeros(m,1); 
v = zeros(m,1);

Out.e  = [];
Out.et = [];
tic;

for i = 1 : max_iter
	
    vm1 = v;
    xm1 = x;
    
	%v-step
	tv = A*x-y-w/rho;
    v  = shrinkage_Lq(tv, 1, 1/lamda, rho);
    
    %x-step
    tao = 0.99; %for orthonornal A
    z = x - tao*(A'*(A*x - y - v - w/rho)); 
    x = shrinkage_Lq(z, q, tao, rho);
    
    
	%u-step
    Ax = A*x;
	w = w - rho*(Ax - y - v);

    Out.e  = [Out.e norm(x-xm1)];
    Out.et = [Out.et toc];
    
    %terminate when both primal and dual residuals are small
    if (norm(rho*(v-vm1))< sqrt(n)*ABSTOL && norm(Ax-y-v)< sqrt(n)*ABSTOL) 
%         break;
    end
end

end
