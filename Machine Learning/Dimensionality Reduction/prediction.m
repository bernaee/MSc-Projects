function prediction(disc_func,validation_0,validation_1,mean_0,cov_0,prior_0,mean_1,cov_1,prior_1)
pred_labels_0 =predict_class(disc_func,validation_0,mean_0,cov_0,prior_0,mean_1,cov_1,prior_1);
pred_labels_1 =predict_class(disc_func,validation_1,mean_0,cov_0,prior_0,mean_1,cov_1,prior_1);
pred_labels = cat(1,pred_labels_0, pred_labels_1);
calc_confusion_matrix(pred_labels_0,pred_labels_1);
end

