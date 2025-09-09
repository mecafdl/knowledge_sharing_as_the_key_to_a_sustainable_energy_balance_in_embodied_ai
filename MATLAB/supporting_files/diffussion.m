clc
close all

alpha = 0.1;
beta1 = 0.4;
beta2 = 0.4;


A = zeros(6);

A(1:3,1:3) = alpha*(~eye(3));
A(1:3,4:6) = beta1/3*ones(3);
A(4:6,4:6) = alpha*(~eye(3));
A(4:6,1:3) = beta2/3*ones(3);

D = diag(sum(A,2));
L = diag(sum(A,2)) - A;

% knowledge = @(n,cj) (1/(1 - exp(-rate)))*(exp(-rate/cj*n) - exp(-rate));

%%
clc
N_S = 6;
N_K = 2;
N   = N_S/N_K;

A = rand(N,N); % generate a random n x n matrix

% construct a symmetric matrix using either
A = 0.5*(A+A');
A = A - diag(diag(A));
A = A + 5*eye(N);

% A =  rand(3);
% A = triu(A) + (triu(A,1))';
% for i=1:3
%     A(i,:) = A(i,:)./sum(A(i,:)); 
% end
% A(2,1) = 0;
% A(1,2) = 0;

% A(2,3) = 0;
% A(3,2) = 0;

%%
close all
clearvars
clc
clc
c0      = 100;
alpha_i = 10;
N_S     = 10;
N_K     = 2;
N       = N_S/N_K;
Spp     = sympositivedefinitefactory(N);

A = Spp.rand();
A = A + abs(min(A,[],'all'));
A = A./(sum(A,'all'))
% A = A - diag(diag(A));
% A = A./(max(A,[],'all'));
% A = 0.3*alpha_i*A + alpha_i*eye(N)
% A = 0.9*alpha_i*A + alpha_i*eye(N)
[V,D] = eig(-A/c0)
c = (V\ones(N,1));
%
close all
n = 0:0.001:c0;
% n=0;
f =  @(n) V*(exp(diag(D)*n).*c);% (V.*transpose(exp(diag(D)*n)))*c;
% f = @(n) c(1)*V(:,1)*exp(D(1,1).*n) + ...
%     c(2)*V(:,2)*exp(D(2,2).*n) + ...
%     c(3)*V(:,3)*exp(D(3,3).*n);
x = f(n);

f = @(t,x) -A*x; 
x = ode4(f,n,sum(A,2));

min(x(:))
figure('Color',[1 1 1])
plot(n,x)
hold on
plot(n,sum(x,2),'k--')
%%
clc
clearvars

% V = [0.2 0.1; 0.6 0.1];
% W = [0.2 0.1;0.1 0.6];
N = 3;
Spp   = sympositivedefinitefactory(N);
W     = Spp.rand();
W     = W + abs(min(W,[],'all'));
W     = W.*(~eye(N)) + eye(N);
W     = W./sum(W,2)

% W = [0.7 0.3;0.3 0.7];
% D = diag([5 5]);
D = diag(rand(N,1));%5*eye(N);
A = W*D*(W^-1)
[V_A,D_A] = eig(A)
c  = (W\sum(W,2));
%
close all
n = 0:0.01:50;
% n=0;
% f =  @(n) V*(exp(diag(D)*n).*c);% (V.*transpose(exp(diag(D)*n)))*c;
% f = @(n) c(1)*V(:,1)*exp(D(1,1).*n) + ...
%     c(2)*V(:,2)*exp(D(2,2).*n) + ...
%     c(3)*V(:,3)*exp(D(3,3).*n);
% x = f(n);

f = @(t,x) -A*x; 
x = ode4(f,n,sum(W,2));



min(x(:))
plot(n,x)
hold on
plot(n,mean(x,2),'k--')
%%

B = [6, 1; 4, 3] 
[V,D] = eig(-B)


%%
clc
close all
A = -[0.6 0.1;0.1 0.2];
n = 0:0.01:50;
f = @(t,x) A*x; 
x = ode4(f,n,-sum(A,2));
plot(n,x)
%           tspan = 0:0.1:20;
%           y = ode4(@vdp1,tspan,[2 0]);  
%           plot(tspan,y(:,1));
%%
clc
close all
n = 0:0.01:100;
x = V*exp(-diag(D).*n).*(V\ones(2,1));
plot(n,x)
%%
A = A + alpha_i*eye(N);
A
eig(A)
%%
N        = 50;
rate     = 2;
n = 0:N;
% bsigma_0 = exp(-(6/N)*n);
bsigma_0 = exp(-0.1*n);
% (exp(-6/cj(j)*n))
figure; plot(n,bsigma_0)
%%
N  = 30;
c0 = 100;
clc
close all
A = (5/c0)*eye(N);
n = 0:0.01:10;
var = -100;
sig = @(x) exp(var*x) ./ (1 + exp(var*x));
plot(n,sig(n-2))
close all


s = [1:N]';
n = 0:1:N*c0;
% f = @(t,x) -A*x + bsigma_0(1:end-1)'.*(t==0); 
f = @(t,x) A*(-x + ...
    sig(abs(x - exp(-(6/N)*(s-1)))).*exp(-(6/N)*(s-1)));

% f = @(t,x) A*(-x + bsigma_0(1:end-1)'); 
% f = @(t,x) A*(ones(N,1) - x); 
% f = @(t,x) A*(bsigma_0(1:end-1)' - x); 
% x = ode4(f,n,[1;1-0.4]);
% x = ode4(f,n,bsigma_0(1:end-1)');
% x = ode4(f,n,zeros(N,1));

% x = ode4(@(t,x) fcn_general_dynamics(t, x, A, N, s),n,zeros(N,1));
x = ode4(@(t,x) fcn_general_dynamics(t, x, A, N, s),n,ones(N,1));
figure
plot(n,x)
figure
plot(x(end,:))


%%
clc
close all
n = 0:0.01:50;
f1 = @(t,x) 0.3*(1 - x); 
f2 = @(t,x) 0.3*(- x) + 1;
f3 = @(t,x) 0.1*(1 - x);
x1 = ode4(f1,n,zeros(N,1));
x2 = ode4(f2,n,zeros(N,1));
x3 = ode4(f3,n,zeros(N,1));
figure
plot(n,x1)
xlim([0 50])
hold on
% plot(n,x2)
plot(n,x3)
legend('0.3','NaN','0.1')