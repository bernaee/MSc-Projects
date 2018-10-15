function [pred_labels] = predict_class(disc_func,validation,mean_0,cov_0,prior_0,mean_1,cov_1,prior_1)
likelihood_0 = calc_discriminant(disc_func,validation, mean_0,cov_0,prior_0);
likelihood_1 = calc_discriminant(disc_func,validation, mean_1,cov_1,prior_1);
for i=1:length(validation)
    if likelihood_0(i) > likelihood_1(i)
        pred_labels(i)=0;
    else
        pred_labels(i)=1;
    end
end
pred_labels = pred_labels.';
end

