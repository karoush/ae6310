clc
clear all 
close all

%%% This file is used to:
%%% 1B- create the contours of the quadratic penalty function
%%% 1D- plot the num of function evals & multiplier est. for log-barrier

% [x, lam]= initial_constrained()
logBarrier_vary()
% quadPenalty_vary() %doesn't work

n= 1000;
x1= linspace(-3.5,2,n);
x2= linspace(-3.5,2,n);
[X1, X2]= meshgrid(x1,x2);

% base_graph(X1,X2)
% quad_graphs(X1,X2)
% log_graphs(X1,X2)

function logBarrier_vary() %use MATLAB
funcEvals= [];
lambda_est= [];
n= 500; 

for mu= linspace(1,0.001,n)
    fun_quad= @(x)(x(1)+2)^2 +10*(x(2)+3)^2 -...
        mu*log(-1*(x(1)^2+ x(2)^2 -2));
    nonlcon = @cons;
    x0= [-1,0];
    A= []; b= [];
    Aeq= []; beq= [];
    lb= []; ub= [];
    [x,fval,exitflag,output] = fmincon(fun_quad,x0,A,b,Aeq,beq,lb,ub,nonlcon);
    funcEvals= [funcEvals, output.funcCount];
    
    c= x(1)^2+ x(2)^2 -2;
    lambda_est= [lambda_est, -mu/c];
end
mu= linspace(1,0.001,n);
figure()
plot(1./mu, funcEvals,'.')
xlabel('mu')
% set(gca, 'XScale', 'log')
set(gca, 'xdir','reverse')
ylabel('Function evals')

figure()
plot(1./mu, lambda_est,'.')
hold on
plot(1./mu, 11.3536.*ones(1,n),'k--')
xlabel('1/mu')
set(gca, 'XScale', 'log')
ylabel('Multiplier estimate')
end

function quadPenalty_vary() %Use Python
funcEvals= [];
lambda_est= [];
n= 500; 

for rho= linspace(1,1000,n)
    fun_quad= @(x)(x(1)+2)^2 +10*(x(2)+3)^2 +...
        (rho/2)*(max(x(1)^2+ x(2)^2 -2,0).^2);
    nonlcon = @cons;
    x0= [-1,-1];
    A= []; b= [];
    Aeq= []; beq= [];
    lb= []; ub= [];
    [x,fval,exitflag,output] = fmincon(fun_quad,x0,A,b,Aeq,beq,lb,ub,nonlcon);
    funcEvals= [funcEvals, output.funcCount];
    
    c= x(1)^2+ x(2)^2 -2;
    lambda_est= [lambda_est, rho*c];
end
rho= linspace(1,1000,n);
figure()
plot(rho, funcEvals,'+')
xlabel('rho')
set(gca, 'XScale', 'log')
ylabel('Function evals')

figure()
plot(rho, lambda_est,'+')
xlabel('rho')
set(gca, 'XScale', 'log')
ylabel('Multiplier estimate')
end

function [x,lam]=initial_constrained()
fun= @(x)(x(1)+2)^2 +10*(x(2)+3)^2;
nonlcon = @cons;
x0= [-1,-1];
A= [];
b= [];
Aeq= [];
beq= [];
lb= [];
ub= [];
[x,fval,exitflag,output] = fmincon(fun,x0,A,b,Aeq,beq,lb,ub,nonlcon);
funcEvals= output.funcCount;
lam=calcLam(x);
end

function lam=calcLam(x)
A_xstar= [-2*x(1), -2*x(2)];
delF= [2*x(1)+4; 20*x(2)+60];
lam= (A_xstar*delF)/(A_xstar*A_xstar');
end

function [c,ceq]= cons(x)
c= x(1)^2+ x(2)^2 -2;
ceq=[];
end

function base_graph(X1,X2)
figure()
F= obj_fun(X1,X2);
contour(X1,X2,F)
hold on 
viscircles([0,0],sqrt(2));
hold off
title('Objective function contours')
xlabel('x1'); ylabel('x2');
end

function quad_graphs(X1,X2)
figure()
rho= 50;
f2= quadPenalty(X1,X2,rho);
contour(X1,X2,f2)
hold on 
viscircles([0,0],sqrt(2));
hold off
title_text= strcat('Quadratic Penalty contours, rho= ', num2str(rho));
title(title_text)
xlabel('x1'); ylabel('x2');

figure()
rho= 1000;
f2= quadPenalty(X1,X2,rho);
contour(X1,X2,f2)
hold on 
viscircles([0,0],sqrt(2));
hold off
title_text= strcat('Quadratic Penalty contours, rho= ', num2str(rho));
title(title_text)
xlabel('x1'); ylabel('x2');
end

function log_graphs(X1,X2)
figure()
mu= 0.5;
f2= logBarrier_penalty(X1,X2,mu);
contour(X1,X2,f2)
hold on 
viscircles([0,0],sqrt(2));
hold off
title_text= strcat('Quadratic Penalty contours, mu= ', num2str(mu));
title(title_text)
xlabel('x1'); ylabel('x2');

figure()
mu= 0.1;
f2= logBarrier_penalty(X1,X2,mu);
contour(X1,X2,f2)
hold on
viscircles([0,0],sqrt(2));
hold off
title_text= strcat('Quadratic Penalty contours, mu= ', num2str(mu));
title(title_text)
xlabel('x1'); ylabel('x2');
end

function f= obj_fun(x1,x2)
f=(x1+2)^2 +10*(x2+3)^2;
end

function f= quadPenalty(x1,x2,rho)
f=(x1+2)^2 +10.*(x2+3)^2;
con_val= x1.^2+x2.^2 -2;
dim= size(x1);
zero= zeros(dim(1));
f=f+ (rho/2)*(max(con_val,zero).^2);
end

function f= logBarrier_penalty(x1,x2,mu)
dim= size(x1);
f= ones(dim(1));
for i=1:1:dim(1) %iterate through x1 (cols)
    for j=1:1:dim(1) %iterate through x2 (rows)
        x1_curr= x1(1,i);
        x2_curr= x2(j,1);
        
        c= x1_curr^2 + x2_curr^2 -2;
        if c<0 %feasible if c <0
           f(i,j)=(x1_curr+2)^2 +10.*(x2_curr+3)^2 +mu*log(-c); 
        else 
           f(i,j)= NaN;
        end
    end
end
f= f';
end
