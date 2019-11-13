function classification(data)

x_t = data(:,1:3);
x_r= data(:,4);

[training_labels,class_0,class_1,training_0,training_1,validation_0,validation_1] = split_train_validation_data(x_t,x_r);

plot3(class_0(:,1),class_0(:,2),class_0(:,3),'+',class_1(:,1),class_1(:,2),class_1(:,3),'*');
title('Scatter Plot of Classes');
legend('class_0','class_1');
saveas(gcf,'plot_1','png');

[mean_0,cov_0]= calc_class_params(training_0);
prior_0 = length(training_labels (training_labels == 0,:))/length(training_labels );
[mean_1,cov_1] =calc_class_params(training_1);
prior_1 = length(training_labels (training_labels == 1,:))/length(training_labels );
cov= prior_0*cov_0 + prior_1*cov_1;

display('Quadratic Discriminant')
prediction('quadratic',validation_0,validation_1,mean_0,cov_0,prior_0,mean_1,cov_1,prior_1);
display('Linear Discriminant')
prediction('linear',validation_0,validation_1,mean_0,cov,prior_0,mean_1,cov,prior_1);
display('Naive Bayes Discriminant')
prediction('naivebayes',validation_0,validation_1,mean_0,cov,prior_0,mean_1,cov,prior_1);
display('Euclidean Discriminant')
prediction('euclidean',validation_0,validation_1,mean_0,cov,prior_0,mean_1,cov,prior_1);

display('PCA')
[mean_t,cov_t]= calc_class_params(x_t);
[eigenvector,eigenvalues] = eig(cov_t);
[max_eigenvalue,max_eigenvector_idx]=max(diag(eigenvalues));
red_x_t=(eigenvector(:,max_eigenvector_idx).'*(x_t -mean_t).').';
[red_training_labels,red_class_0,red_class_1,red_training_0,red_training_1,red_validation_0,red_validation_1] = split_train_validation_data(red_x_t,x_r);
plot(red_class_0(:,1),zeros(1,100),'+',red_class_1(:,1),zeros(1,100),'*');
title('Scatter Plot of Classes After Applying PCA')
legend('class_0','class_1');
saveas(gcf,'plot_2','png');
[red_mean_0,red_cov_0]= calc_class_params(red_training_0);
red_prior_0 = length(red_training_labels(red_training_labels == 0,:))/length(red_training_labels );
[red_mean_1,red_cov_1] =calc_class_params(red_training_1);
red_prior_1 = length(red_training_labels (red_training_labels == 1,:))/length(red_training_labels );
red_cov= red_prior_0*red_cov_0 + red_prior_1*red_cov_1;
display('Quadratic Discriminant')
prediction('naivebayes',red_validation_0,red_validation_1,red_mean_0,red_cov_0,red_prior_0,red_mean_1,red_cov_1,red_prior_1);
display('Linear Discriminant')
prediction('linear',red_validation_0,red_validation_1,red_mean_0,red_cov,red_prior_0,red_mean_1,red_cov,red_prior_1);
display('Naive Bayes Discriminant')
prediction('naivebayes',red_validation_0,red_validation_1,red_mean_0,red_cov,red_prior_0,red_mean_1,red_cov,red_prior_1);
display('Euclidean Discriminant')
prediction('euclidean',red_validation_0,red_validation_1,red_mean_0,red_cov,red_prior_0,red_mean_1,red_cov,red_prior_1);

display('LDA')
w = (inv(cov_0 + cov_1)*(mean_0-mean_1).');
z_t = (w.'*x_t.').';
[z_training_labels,z_class_0,z_class_1,z_training_0,z_training_1,z_validation_0,z_validation_1] = split_train_validation_data(z_t,x_r);
plot(z_class_0(:,1),zeros(1,100),'+',z_class_1(:,1),zeros(1,100),'*');
title('Scatter Plot of Classes After Applying LDA')
legend('class_0','class_1');
saveas(gcf,'plot_3','png');
[z_mean_0,z_cov_0]= calc_class_params(z_training_0);
z_prior_0 = length(z_training_labels(z_training_labels == 0,:))/length(z_training_labels );
[z_mean_1,z_cov_1] =calc_class_params(z_training_1);
z_prior_1 = length(z_training_labels(z_training_labels == 1,:))/length(z_training_labels );
z_cov= z_prior_0*z_cov_0 + z_prior_1*z_cov_1;
display('Quadratic Discriminant')
prediction('quadratic',z_validation_0,z_validation_1,z_mean_0,z_cov_0,z_prior_0,z_mean_1,z_cov_1,z_prior_1);
display('Linear Discriminant')
prediction('linear',z_validation_0,z_validation_1,z_mean_0,z_cov,z_prior_0,z_mean_1,z_cov,z_prior_1);
display('Naive Bayes Discriminant')
prediction('naivebayes',z_validation_0,z_validation_1,z_mean_0,z_cov,z_prior_0,z_mean_1,z_cov,z_prior_1);
display('Euclidean Discriminant')
prediction('euclidean',z_validation_0,z_validation_1,z_mean_0,z_cov,z_prior_0,z_mean_1,z_cov,z_prior_1);

end


