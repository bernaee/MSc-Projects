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

%calculate discriminant
discriminant_1 = log(likelihood_1) + log(prior_1);
discriminant_2 = log(likelihood_2) + log(prior_2);

%calculate risk
risk_1 = loss_1_2*(posterior_2);
risk_2 = loss_2_1*(posterior_1);

[lb_1, lb_2] =find_decision_boundary(data,likelihood_1,likelihood_2);
[pb_1, pb_2] = find_decision_boundary(data,posterior_1,posterior_2);
[rb_1, rb_2] =find_decision_boundary(data,risk_1,risk_2);
[db_1, db_2] =find_decision_boundary(data,discriminant_1,discriminant_2);

row_names = {'Class_Boundary_1';'Class_Boundary_2'};
Likelihood = [lb_1; lb_2] ;
Posterior = [pb_1; pb_2];
Risk = [rb_1; rb_2];
Discriminant = [db_1; db_2];
T = table(Likelihood,Posterior,Risk,Discriminant,'RowNames',row_names);
display(T)


%calculate confusion matrix
if loss_rej==0
    [tp,s] = size(labels(labels(risk_2 > risk_1) == 1));
    [fp,s] = size(labels(labels(risk_2 > risk_1) == -1));
    [fn,s] = size(labels(labels(risk_1 > risk_2) == 1));
    [tn,s] = size(labels(labels(risk_1 > risk_2) == -1)); 
    
    row_names = {'Actual_Pos';'Actual_Neg'};
    Pred_Pos = [tp;fp];
    Pred_Neg = [fn;tn];
    T = table(Pred_Pos,Pred_Neg,'RowNames',row_names)
    
else
    risk_reject =  loss_rej*(posterior_2) + loss_rej*(posterior_1); %loss_rej
    [tp,s] = size(labels(labels((risk_1 < risk_2)  & (risk_1 < loss_rej)) == 1));
    [fp,s] = size(labels(labels((risk_1 < risk_2)  & (risk_1 < loss_rej)) == -1));
    [fn,s] = size(labels(labels((risk_2 < risk_1)  & (risk_2 < loss_rej)) == 1));
    [tn,s] = size(labels(labels((risk_2 < risk_1)  & (risk_2 < loss_rej)) == -1)); 
    [rej_p,s] = size(labels(labels((risk_1 < risk_2)  & (risk_1 > loss_rej)) == 1));
    [rej_n,s] = size(labels(labels((risk_2 < risk_1)  & (risk_2 > loss_rej)) == -1)); 
    
     row_names = {'Actual_Pos';'Actual_Neg'};
    Pred_Pos = [tp;fp];
    Pred_Neg = [fn;tn];
    Rejection = [rej_p;rej_n];
    T = table(Pred_Pos,Pred_Neg,Rejection,'RowNames',row_names)
    
end

if isDisplay==1  
    % plot likelihoods
    figure
    subplot(4,1,1)
    plot(data,likelihood_1, 'g.', data,likelihood_2 ,'b.' )
    hold on
    plot(lb_1, likelihood_1(find(data == lb_1)),'r*')
    plot(lb_2, likelihood_1(find(data == lb_2)),'r*')
    title('Likelihoods')
    xlabel('x')
    ylabel('p(x|Ci)')

    % plot posteriors    
    subplot(4,1,2)
    plot(data,posterior_1,'g.',data,posterior_2,'b.')
    hold on
    plot(pb_1, posterior_1(find(data == pb_1)),'r*')
    plot(pb_2, posterior_2(find(data == pb_2)),'r*')
    title('Posteriors')
    xlabel('x')
    ylabel('p(Ci|x)')
    
    % plot risk
    subplot(4,1,3)
    plot(data,risk_1,'g.',data,risk_2,'b.')
    hold on
    plot(rb_1, risk_1(find(data == pb_1)),'r*')
    plot(rb_2, risk_2(find(data == pb_2)),'r*')
    title('Risk')
    xlabel('x')
    ylabel('R(ai|x)')

    % plot discriminants
    subplot(4,1,4)
    plot(data,discriminant_1,'g.',data,discriminant_2,'b.')
    hold on
    plot(db_1, discriminant_1(find(data == pb_1)),'r*')
    plot(db_2, discriminant_2(find(data == pb_2)),'r*')
    title('Discriminant')
    xlabel('x')
    ylabel('gi(x)')
    
    
    

end
end

