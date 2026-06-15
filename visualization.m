% Single Script to perform visualizations of GPU computations
tic;
N=128;        %user changes the N value here
M1=5;tfinal=10;dt = 0.01; 
a=0;b=2*pi;
CH3D_GPU_project_2026_BCP(dt,M1,tfinal,N)    %comment out regular or nested
%CH3D_GPU_project_2026_Nested(dt,M1,tfinal,N) 

h=(b-a)/N;
load('U_numerical','U');
x=[a:h:b-h]; 
[X,Y,Z] = meshgrid(x,x,x);
figure(500)
isosurface(X,Y,Z,U,-.3)
isosurface(X,Y,Z,U,-.15);
isosurface(X,Y,Z,U,-.05);
isosurface(X,Y,Z,U,.05);
isosurface(X,Y,Z,U,.15);
isosurface(X,Y,Z,U,.3)

ax = gca; 
ax.FontSize = 14;
camlight; lighting phong
axis([a b a b a b])
%title(['BHM method, Time = ' num2str(tfinal) ' , dt = ' num2str(dt)] ); 
title(['Time = ' num2str(tfinal)] ); 

a=toc;
minutes=a/60;hours=a/60^2;
minutes_hours=[minutes hours]
