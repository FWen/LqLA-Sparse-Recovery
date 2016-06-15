clear all; clc;

WIDTH = 256;
n = WIDTH^2;   	% signal dimension
m = round(0.4*n);    	% number of measurements

XX(:,:,1) = phantom('Modified Shepp-Logan',WIDTH);
XX(:,:,2) = imresize(double(imread('MRI.jpg')),[WIDTH WIDTH]);

J = randperm(n); J = J(1:m);    % m randomly chosen indices
A = partialDCT(n,m,J); 

figure(1);subplot(121);imagesc(XX(:,:,1));
set(gca,'ytick',[]);set(gca,'xtick',[]);box off;title('Shepp-Logan');
subplot(122);imagesc(XX(:,:,2));
set(gca,'ytick',[]);set(gca,'xtick',[]);box off;title('MRI');

xx =[];
for iX=1:size(XX,3)
    X = XX(:,:,iX);
    xxm1 = xx;

    %Obtain wavelet coeffs
    [C,S] = wavedec2(X, 3, 'haar');
    Norm_C = norm(C);
    x = C'/Norm_C;

    y0 = A*x;
    
    % --Two-component Gaussian mixture noise------
    SNR=20;
    epsilon = 0.1; kappa=1000;
    noise = gmrnd([0 0],[1 sqrt(kappa)],[1-epsilon epsilon],m,1);
    noise = noise/std(noise) *10^(-SNR/20)*std(y0);    
        
    y = y0 + noise;

    t0 = tic;

    lamda_min = 1e-5; lamda_max = 1e0;
    lamdas    = logspace(log10(lamda_min), log10(lamda_max),40);
    for k = 1:length(lamdas); 
        [YALL1 Out1] = YALL1_admm(A, y, lamdas(k), x);
        relerr(3,k)  = norm(YALL1 - x)/norm(x);
        xx(:,k,3)    = YALL1;
    end
    [mv mi] = min(relerr(1,:));
    x_0 = xx(:,mi,3); 


    for k = 1:length(lamdas)
        [iX k]

        % L1LS-FISTA (q=1)
        [L1LS Out2] = lq_l2_fista(A, y, lamdas(k), 1, x);
        relerr(1,k) = norm(L1LS - x)/norm(x);
        xx(:,k,1)   = L1LS;

        % LqLS-ADMM (q=0.5)
        [LqL2_05 Out3] = lq_l2_admm(A, y, lamdas(k), 0.5, x, x_0);
        relerr(2,k)    = norm(LqL2_05 - x)/norm(x);
        xx(:,k,2)      = LqL2_05;
        
        % LqLA-ADMM (q=0.2)
        [LqL1_02 Out4] = lq_l1_admm(A, y, lamdas(k), 0.2, x, x_0);
        relerr(4,k)    = norm(LqL1_02 - x)/norm(x);
        xx(:,k,4)      = LqL1_02;
        
        % LqLA-ADMM (q=0.5)
        [LqL1_05 Out5] = lq_l1_admm(A, y, lamdas(k), 0.5, x, x_0);
        relerr(5,k)    = norm(LqL1_05 - x)/norm(x);
        xx(:,k,5)      = LqL1_05;
        
        % LqLA-ADMM (q=0.7)
        [LqL1_07 Out6] = lq_l1_admm(A, y, lamdas(k), 0.7, x, x_0);
        relerr(6,k)    = norm(LqL1_07 - x)/norm(x);  
        xx(:,k,6)      = LqL1_07;

    end
    RelErr(:,:,iX) = relerr;

    figure(3);semilogy(lamdas,relerr(1,:),'r-',lamdas,relerr(2,:),'r--',lamdas,relerr(3,:),'r:',...
        lamdas,relerr(4,:),'b-',lamdas,relerr(5,:),'b--',lamdas,relerr(6,:),'b:','linewidth',2);grid off;set(gca,'xscale','log');grid;
    toc(t0)

    Method_Name = char('L1LS-FISTA','LqLS-ADMM (q=0.5)','YALL1','LqLA-ADMM (q=0.2)','LqLA-ADMM (q=0.5)','LqLA-ADMM (q=0.7)');

    figure(4);   
    for ix=1:size(xx,3)
        [mv mi] = min(relerr(ix,:));
        x_recs(:,ix) = xx(:,mi,ix);   

        X_rec(:,:,ix) = waverec2(x_recs(:,ix)*Norm_C, S, 'haar');
        PSNR(ix)     = psnr(X, X_rec(:,:,ix));
        subplot(4,3,ix+(iX-1)*size(xx,3));imagesc(X_rec(:,:,ix));
        set(gca,'ytick',[]);set(gca,'xtick',[]);box off;
        title([deblank(Method_Name(ix,:)) ', ', num2str(PSNR(ix), '%10.2f'), ' dB']);
    end        
end