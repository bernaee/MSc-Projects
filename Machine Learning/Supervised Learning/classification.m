function classification(data_set,loss_1_2,loss_2_1,loss_rej, isDisplay)

data = data_set(:,1);
labels= data_set(:,2);
data_size = size(data,1);
class_1 =  data(labels == 1);
class_2 =  data(labels == -1);

% calculate class 1 parameters
[mean_1,var_1] =calc_class_params(class_1);
prior_1 = size(class_1,1)/data_size;
likelihood_1=calc_likelihood(data,mean_1,var_1);

% calculate class 2 parameters
[mean_2,var_2] =calc_class_params(class_2);
prior_2 = size(class_2,1)/data_size;
likelihood_2=calc_likelihood(data,mean_2,var_2);

%calculate evidence and posteriors
evidence =(likelihood_1.*prior_1 + likelihood_2.*prior_2);
posterior_1 = (likelihood_1.* prior_1)./ evidence;
posterior_2 = (likelihood_2.* prior_2)./ evidence;

%calculate risk
risk_1 = loss_1_2*(1-posterior_1);
risk_2 = loss_2_1*(1-posterior_2);

%calculate discriminant
discriminant_1 = log(likelihood_1) + log(prior_1);
discriminant_2 = log(likelihood_2) + log(prior_2);

%calculate confusion matrix
if loss_rej==0
    class_1_risk = loss_1_2 / (loss_1_2 +loss_2_1)
    tp = size(labels(labels(posterior_1 >class_1_risk ) == 1));
    fp = size(labels(labels(posterior_1 >class_1_risk ) == -1))
    fn = size(labels(labels(posterior_1 < class_1_risk) == 1))
    tn = size(labels(labels(posterior_1 < class_1_risk) == -1)) 
else
    class_1_lower_risk = loss_rej /loss_2_1
    class_1_upper_risk = (loss_1_2 -loss_rej) / loss_1_2
    tp = size(labels(labels(posterior_1 > class_1_upper_risk) == 1))
    fp = size(labels(labels(posterior_1 > class_1_upper_risk ) == -1))
    fn = size(labels(labels(posterior_1 < class_1_lower_risk) == 1))
    tn = size(labels(labels(posterior_1 < class_1_lower_risk) == -1)) 
    rej_p = size(labels(labels(posterior_1 < class_1_upper_risk & posterior_1 > class_1_lower_risk) == 1))
    rej_n = size(labels(labels(posterior_1 < class_1_upper_risk & posterior_1 > class_1_lower_risk) == -1)) 
end

if isDisplay==1
    % plot likelihoods
    figure
    subplot(4,1,1)
    plot(data,likelihood_1,data,likelihood_2)
    title('Likelihoods')
    xlabel('x')
    ylabel('p(x|Ci)')
    likelihood_boundary =find_decision_boundary(data,likelihood_1,likelihood_2)


    % plot posteriors
    subplot(4,1,2)
    plot(data,posterior_1,data,posterior_2)
    title('Posteriors')
    xlabel('x')
    ylabel('p(Ci|x)')
    posterior_boundary = find_decision_boundary(data,posterior_1,posterior_2)

    % plot risk
    subplot(4,1,3)
    plot(data,risk_1,data,risk_2)
    title('Risk')
    xlabel('x')
    ylabel('R(ai|x)')
    risk_boundary =find_decision_boundary(data,risk_1,risk_2)

    % plot discriminants
    subplot(4,1,4)
    plot(data,discriminant_1,data,discriminant_2)
    title('Discriminant')
    xlabel('x')
    ylabel('gi(x)')
    discriminant_boundary =find_decision_boundary(data,discriminant_1,discriminant_2)

end
end

