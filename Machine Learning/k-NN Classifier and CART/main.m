cls_data= load('C:\Users\Bernisko\Google Drive\Masters\BM 59D\HW4\BM59D_Hw3_Data.mat');
% X = cls_data.X;
% display('X')
% run_kNN(X)
% display('Y')
% Y = cls_data.Y;
% run_kNN(Y)

reg_data= load('C:\Users\Bernisko\Google Drive\Masters\BM 59D\HW4\BM59D_Hw4_Data.mat');
training.x=reg_data.x_tr;
training.r=reg_data.r_tr;
validation.x=reg_data.x_val;
validation.r=reg_data.r_val;
run_decision_tree(training,validation, [0.5,0.2,0.05])