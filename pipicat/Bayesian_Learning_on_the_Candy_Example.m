%% This is an auxiliary code for blog
% "Bayesian Learning on the Candy Example"
% 12/20/17
% Yi DING


%% Bayesian updating
clc;clear;
p_h = [0.1;0.2;0.4;0.2;0.1];
p_lime = [0;0.25;0.5;0.75;1];

num_candy_drawn = 10;
% Inintialization
p_h_d = zeros(num_candy_drawn+1,5);
% Updating
for i = 0:num_candy_drawn
    p_d_h = (p_lime.^i);
    p_d = p_h'* p_d_h;
    p_h_d(i+1,:) = (p_h.*p_d_h)'/p_d;
end

%% Prediction
p_next_is_lime = zeros(num_candy_drawn,1);
for i = 0:num_candy_drawn
    p_next_is_lime(i+1) = p_lime'*p_h_d(i+1,:)';
end

%% Plot
figure
ftsize = 20;
plot(0:10,p_h_d,'LineWidth',2);
set(gca,'fontsize',ftsize);
grid on;
title('Bayesian updating for the candy bag example','FontSize',ftsize)
xlabel(['\# of samples in ','$\mathbf{d}$'],'FontSize',ftsize,...
    'interpreter', 'latex');
ylabel('Posterior probability of hypothesis','FontSize',ftsize)
leg1 = legend('$P(h_1|d)$','$P(h_2|d)$','$P(h_3|d)$','$P(h_4|d)$',...
    '$P(h_5|d)$');
set(leg1,'Interpreter','latex','FontSize',ftsize);

figure
plot(0:10,p_next_is_lime,'LineWidth',2);
set(gca,'fontsize',ftsize);
grid on;
title('Bayesian prediction','FontSize',ftsize)
xlabel(['\# of samples in ','$\mathbf{d}$'],'FontSize',ftsize,...
    'interpreter', 'latex');
ylabel(['$P($','next candy is lime','$|\mathbf{d})$'],'FontSize',ftsize,...
    'interpreter', 'latex')

