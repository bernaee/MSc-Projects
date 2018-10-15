data= load('C:\Users\Bernisko\Google Drive\Masters\BM 59D\HW2\BM59D_Hw2_Data.mat')

% CLASSIFICATION
xtr_classification = data.xtr_classification;
xval_classification = data.xval_classification;
display('The Plots and The Confusion Matrix of Training and Validation Set')
classification(xtr_classification,1,1,0,0)
classification(xval_classification,1,1,0,0)
display('The Confusion Matrix of Training and Validation Set With Asymmetric Loss')
classification(xtr_classification,1/2,1,0,0)
classification(xval_classification,1/2,1,0,0)
display('The Confusion Matrix of Training and Validation Set With Asymmetric Loss and Rejection')
classification(xtr_classification,1/2,1,0.2,0)
classification(xval_classification,1/2,1,0.2,0)



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
    plot(errors(i,:));
    title(sprintf('The Sample-%d Errors',i));
    xlabel('Degree');
    ylabel('LSE');
end
printmat(errors, 'Error Matrix', 'Sample1 Sample2 Sample3', '1 2 3 4 5 6 7 8 9' )