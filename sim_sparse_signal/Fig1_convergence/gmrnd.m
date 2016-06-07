function [x,m,v] = gmrnd(varargin)
%[x, m, v] = gmrnd(mu, sigma, weight, m, n ...);
%[x, m, v] = gmrnd(mu, sigma, weight, [m, n, ...]);
%
%Generate Gaussian mixture random data (m x n...).  The distribution
%is a sum of weighted Gaussian distributions with means `mu' and standard
%deviations `sigma'.  The weightings are given by `weight'.  `m' and
%`v' are the theoretical mean and variance of the Gaussian mixture.

%Hwa-Tung Ong, 18 Aug 97

% Retrieve input arguments
mu     = varargin{1}(:);
sigma  = varargin{2}(:);
weight = varargin{3}(:);
weight = weight/sum(weight);
w      = cumsum(weight);
if nargin == 4	% dimensions given by row vector
	n = varargin{4};
	if length(n) == 1 % assume square matrix for single dimension
		n = [n n];
	end
else		% dimensions given by scalars
	flag = 0;
	n = zeros(1,nargin-3);
	for i = 4:length(varargin)
		n(i-3) = varargin{i}(1);
		flag = flag + (length(varargin{i}(:)) > 1);
	end
	if flag warning('Dimension input arguments must be scalar.'); end
end

% Generate Gaussian mixture
p = rand(n);
k = ones(n);
for i = 1:length(mu)-1
	j = find(p >= w(i));
	k(j) = k(j) + 1;	
end

x = reshape(sigma(k),n).*randn(n)+reshape(mu(k),n);

if nargout > 1
	m = weight'*mu;
	v = weight'*(sigma.^2+mu.^2) - m.^2;
end

