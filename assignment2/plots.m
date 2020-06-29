clc 
clear all
close all

n=1000;
x= linspace(-2,2,n);
y= linspace(-2,2,n);
[x1,x2] = meshgrid(x,y);

% f1=zeros(n); f2= zeros(n); for i= linspace(1,n,n)
%     for j=linspace(1,n,n)
%         f1(i,j)= -10*x1(i)^2 +10*x2(j)^2 +4*sin(x1(i)*x2(j)) -2*x1(i)
%         +4*x1(i)^4; f2(i,j)= 100*(x2(j)-x1(i).^2).^2 +(1-x1(i)).^2;
%     end
% end
f1= -10.*x1.^2 +10.*x2.^2 +4*sin(x1.*x2) -2.*x1 +4.*x1.^4;
f2= 100.*(x2-x1.^2).^2 +(1-x1).^2;

[r,c]=find(f1==min(min(f1)));
minx= x(r);
miny= y(c);

figure()
contour(x1,x2,f1)
xlabel('x1'); ylabel('x2'); 
title('Contour Plot of F1')
figure()
mesh(x1,x2,f1)
xlabel('x1'); ylabel('x2'); 
title('3D Plot of F1')

figure()
contour(x1,x2,f2)
xlabel('x1'); ylabel('x2'); 
title('Contour Plot of F2')
figure()
mesh(x1,x2,f2)
xlabel('x1'); ylabel('x2'); 
title('3D Plot of F2')