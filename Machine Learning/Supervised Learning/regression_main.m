data= load('BM59D_Hw2_Data.mat')

%POLYNOMIAL REGRESSION
x_regression= data.x_regression;
x = x_regression(:,1:3);
y = x_regression(:,4);
variances=[0.5,0.3,0.1];
[sample,dimension]=size(x);
degree=9;
for i=1:dimension
    for j=1:degree
       [model,error] =  polynomial_regression(x(:,i),y,variances(i),j); 
       models(:,j,i) = model;
       errors(i,j) = error;
    end
end

degrees = [1 2 3 4 5 6 7 8 9]
figure
for i=1:dimension
    subplot(3,1,i)
    plot(degrees, errors(i,:))
    title(sprintf('The Sample-%d Errors',i));
    xlabel('Degree');
    ylabel('LSE');
end


   
printmat(errors, 'Error Matrix', 'Sample1 Sample2 Sample3', '1 2 3 4 5 6 7 8 9' )