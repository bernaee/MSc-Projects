cls_data= load('BM59D_Hw3_Data.mat');
X = cls_data.X;
Y = cls_data.Y;
run_svm_examples(X, 'X')
run_svm_examples(Y, 'Y')


