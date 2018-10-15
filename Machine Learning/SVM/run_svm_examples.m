function run_svm_examples(data, plot_title)
[training,validation] = split_train_validation_data(data)
[training,validation] = scale(training,validation)
display('Linear Kernel')
[best_cost_0,best_gamma_0]=svm(training,validation,[-5,-4,-3,-2,-1,0,1,2,3,4,5] ,[0], 0,plot_title)
display('Polynomial Kernel')
[best_cost_1,best_gamma_1]=svm(training,validation,[-5,-4,-3,-2,-1,0,1,2,3,4,5] ,[-5,-4,-3,-2,-1,0,1,2,3,4,5], 1,plot_title)
display('RBF Kernel')
[best_cost_2,best_gamma_2]=svm(training,validation,[-5,-4,-3,-2,-1,0,1,2,3,4,5] ,[-5,-4,-3,-2,-1,0,1,2,3,4,5], 2,plot_title)
end

