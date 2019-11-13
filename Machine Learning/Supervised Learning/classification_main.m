data= load('BM59D_Hw2_Data.mat')


% CLASSIFICATION
xtr_classification = data.xtr_classification;
xval_classification = data.xval_classification;
display('The Plots and The Confusion Matrix of Training Set')
classification(xtr_classification,1,1,0,0)
display('The Plots and The Confusion Matrix of Validation Set')
classification(xval_classification,1,1,0,0)
display('The Confusion Matrix of Training Set With Asymmetric Loss')
classification(xtr_classification,1/2,1,0,1)
display('The Confusion Matrix of Validation Set With Asymmetric Loss')
classification(xval_classification,1/2,1,0,0)
display('The Confusion Matrix of Training Set With Asymmetric Loss and Rejection')
classification(xtr_classification,1/2,1,0.2,0)
display('The Confusion Matrix of Validation Set With Asymmetric Loss and Rejection')
classification(xval_classification,1/2,1,0.2,0)

