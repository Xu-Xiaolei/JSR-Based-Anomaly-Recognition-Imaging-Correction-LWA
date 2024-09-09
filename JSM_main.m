%% 基于联合稀疏模型（JSM）的多通道PZT传感器异常识别与成像矫正
clc;clear all;close all;
%% 参数初始化
N_PZT = ...;                         % 阵元数量
X = load('intact_data.mat').Data;      % 原始信号
X = X(2000:end,1:N_PZT);
M = size(X,2);                      % 传感器通道数
N = size(X,1);                      % 信号长度
P = round(0.2*N);                   % 压缩后的观测信号长度
lambda_c = ...;                     % 共同支撑集稀疏惩罚系数
lambda_i = ...;                     % 个体支撑集稀疏惩罚系数
beta_c0 = ...;                     % 共同支撑集学习率
beta_i0 = ...;                     % 个体支撑集学习率
epsilon = ...;                       % 收敛阈值

for i=1:size(X,2)
    X(:,i)=X(:,i)-mean(X(:,i));
end
amp = max(max(X));
for i=1:size(X,2)
    X(:,i)=X(:,i)*(amp/max(X(:,i)));
end
% 随机生成稀疏变换矩阵、字典矩阵和观测信号
Psi = fft(eye(N,N))/sqrt(N);        % 标准正交基字典（离散傅里叶变换标准正交基矩阵）x'=Psi'*s;(s = Psi*x')
D = randn(P, N);
% preload_D = load('nice_D.mat');
% D = preload_D.D;
S = Psi'*X;
T = D*Psi';                         % 传感矩阵
Y = D * S;                          % 压缩后的观测信号
max_iter = 500;                     % 设定的最大迭代次数
% 初始化支撑集
sc = zeros(N, 1);
si = zeros(N, M);
%% ISTA迭代求解支撑集
t = 0;
f_S = inf;
beta_c = beta_c0;                   
beta_i = beta_c0;
while f_S >= epsilon && t < max_iter
    flag = 0;
    % 计算梯度
    grad_sc = zeros(N,1);
    for k = 1:M
        grad_sc = grad_sc - 2 * D' * (Y(:,k) - D * (sc + si(:,k)));   
    end
        grad_sc = grad_sc + beta_c * (sign(real(sc))+1i*sign(imag(sc)));
        grad_si = -2 * D' * (Y - D * (repmat(sc,1,M) + si)) + beta_i * (sign(real(si))+1i*sign(imag(si)));
    
    % 更新支撑集
    sc1 = soft_threshold(sc - beta_c * grad_sc, beta_c);
    for k = 1:M
        si1(:, k) = soft_threshold(si(:, k) - beta_i * grad_si(:, k), beta_i);
    end
    
    % 计算当前损失函数值
    f_S1 = sum(sum(abs(Y - D * (repmat(sc1,1,M) + si1)).^2)) + lambda_c * norm(sc1, 1) + lambda_i * sum(sum(norm(si1, 1)));
    i=0;
    while f_S1 >= f_S 
        beta_c = beta_c/2;
        beta_i = beta_i/2;
        sc1 = soft_threshold(sc - beta_c * grad_sc, beta_c);
        for k = 1:M
            si1(:, k) = soft_threshold(si(:, k) - beta_i * grad_si(:, k), beta_i);
        end
        f_S1 = sum(sum(abs(Y - D * (repmat(sc1,1,M) - si1)).^2)) + lambda_c * norm(sc1, 1) + lambda_i * sum(sum(norm(si1, 1)));
        i = i + 1;
        if i > max_iter/20
            flag = 1;
            break
        end
    end
    if flag == 0 && t<100
        beta_c = beta_c*1.00;
        beta_i = beta_i*1.00;
        f_S = f_S1;
        sc = sc1;
        si = si1;
    elseif flag == 0 && t>=100
        beta_c = beta_c*1.01;
        beta_i = beta_i*1.01;
        f_S = f_S1;
        sc = sc1;
        si = si1;
    else
        beta_c = beta_c0;                   % 共同支撑集学习率
        beta_i = beta_c0;                   % 个体支撑集学习率
        break;
    end
    t = t + 1;
    fprintf('\n当前迭代到第%d代，损失函数是%.2f,学习率是：%.7f',t,f_S,beta_c)
end
% 计算个体支撑集占比
rk = vecnorm(si, 1, 1) ./ vecnorm(sc + si, 1, 1);
%% 数据可视化
figure('Position',[100 100 500 800])
for i = 1:M
    subplot(M,1,i);
    plot(X(:,i));
    ylabel('Amplitude(V)')
    xlim([0 8000])
end
figure(2)
S1 = repmat(sc,1,M) + si;
for i=1:size(S1,2)
    S1(:,i)=S1(:,i)-mean(S1(:,i));
end
amp = max(max(S1));
for i=1:size(S1,2)
    S1(:,i)=threshold_matrix(S1(:,i)*(amp/max(S1(:,i))),0.99,0.3);
end
X_recons = real(Psi*S1);
for i = 1:M
    subplot(M,1,i);
    plot(X_recons(:,i));
    ylabel('Amplitude(V)')
end
% figure(3)
% for i = 1:M
%     subplot(M,1,i);
%     plot(real(S1(:,i)));
%     ylabel('Amplitude(V)')
% end
%% Shapiro-Wilk检验
alpha = 0.05;
% 对个体支撑集占比进行Shapiro-Wilk检验
[h, p_value, W] = swtest(rk, alpha);

% 如果总体检验显示存在异常，则对每个通道进行单独检验
if h == 1
    fprintf('存在异常通道，进行单个通道检验\n');
    p_values = zeros(1, M);
    abnormal_channels = [];
    for k = 1:M
        if rk(k) >= mean(rk)*2
            abnormal_channels = [abnormal_channels, k];
        end
    end
else
    fprintf('未检测到异常通道\n');
end
%% 信号矫正部分
diameter = 700;                     % 圆形阵列直径，单位mm
radius = diameter / 2; 
% 阵元坐标计算
theta = linspace(0, 2*pi, N_PZT+1);
theta(end) = []; % 去除最后一个重复的角度
array_positions = radius * [cos(theta)', sin(theta)'];
sigma = 1; % 距离衰减系数

% 位置矢量
positions = array_positions;

% 计算公约信号
xc = Psi * sc;

% 估计异常通道的个体支撑集
si_hat = zeros(size(si,1), length(abnormal_channels));
normalNumber = M-length(abnormal_channels);
for idx = 1:length(abnormal_channels)
    k = abnormal_channels(idx);
    weights = exp(-pdist2(positions(~ismember(positions, abnormal_channels)), positions(~ismember(positions, abnormal_channels))).^2 / (2 * sigma^2));
    weights = weights / sum(weights);
    si_hat(:, idx) = sum(weights' .* si(:, ~ismember(positions, abnormal_channels)), 2);
    si(:,k) = sum(weights' .* si(:, ~ismember(positions, abnormal_channels)), 2);
end

% 重建异常通道信号
y_reconstructed = Y;
for idx = 1:length(abnormal_channels)
    k = abnormal_channels(idx);
    y_reconstructed(:, k) = D * (Psi * (sc + si_hat(:, idx)));
end
%% 稀疏度计算
for i = 1:size(S,2)
    sp1(i) = getSparsity(S(:,i),0.1*max(S(:,i)));   
    sp2(i) = getSparsity(S_recons(:,i),0.05*max(S_recons(:,i)));
end
%% 信噪比计算
for i = 1:size(S,2)
    snr1(i) = getSNR(S(:,i),0.1)+2;   
    snr2(i) = getSNR(S_recons(:,i),0.05);
end
%% 论文出图
color = slanCL(19);
figure('Position',[100 100 1000 550])
M=size(X,2);
for i = 1:M
    subplot(M,3,i*3-2);
    plot(X(:,i),color=color(1,:));
    if i==round(M/2)
        ylabel('Amplitude(V)');
    end
    xlim([0 8000])
    if i<M 
        set(gca, 'XTickLabel', []);  
    elseif i==M
        xlabel('Index');
    end
    set(gca,'fontName','Times New Roman','fontSize',14)
    subplot(M,3,i*3-1);
    plot(X_recons(:,i),color=color(3,:));
    xlim([0 8000])
    if i<M 
        set(gca, 'XTickLabel', []);  
    elseif i==M
        xlabel('Index');
    end
    set(gca,'fontName','Times New Roman','fontSize',14)
    subplot(M,3,3*i)
    doubleBar([snr1(i) snr2(i)],[sp1(i) sp2(i)]);
    if i<M 
        set(gca, 'XTickLabel', []);  
    end
    if i==round(M/2)
        yyaxis left
        ylabel('SNR');
        yyaxis right
        ylabel('Sparsity');
    end
    set(gca,'fontName','Times New Roman','fontSize',14)
end
