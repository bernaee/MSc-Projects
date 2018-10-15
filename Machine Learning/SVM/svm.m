function [best_cost,best_gamma] = svm(training,validation,cost ,gamma, kernel, plot_title)
best_cost = 0;
best_gamma = 0;
best_acc = 0;
for i = 1:length(cost)
    for j = 1:length(gamma)
        c = cost(i);
        g = gamma(j);
        if kernel==0
            cmd = ['-s 0 -q',' -t ', num2str(kernel),' -c ',num2str( 10^c )];
        else
            cmd = ['-s 0 -q',' -t ', num2str(kernel),' -c ',num2str( 10^c ),' -g ',num2str( 2^g )];
        end
        model = svmtrain(training.label, training.sample, cmd);
        [predict_label, accuracy, dec_values ] = svmpredict( validation.label, validation.sample, model);
        if accuracy(1) > best_acc
            best_model = model;
            best_predict_label = predict_label;
            best_acc = accuracy(1);
            best_cost= c;
            best_gamma = g;
        end
    end
end

best_cost= c
best_gamma = g
tp = length(find(validation.label(find(validation.label== best_predict_label ))==0));
fn =  length(find(validation.label(find(validation.label~= best_predict_label ))==0));
fp =  length(find(validation.label(find(validation.label~= best_predict_label ))==1));
tn =  length(find(validation.label(find(validation.label== best_predict_label ))==1));
cm=[[tp,fn];[fp,tn]]
class_0= validation.sample(best_predict_label == 0,:);
class_1= validation.sample(best_predict_label == 1,:);
if kernel==0

    labels = (best_model.sv_coef.'*full(best_model.SVs))*validation.sample(:,:).' > best_model.rho;
    support_vector_machines = validation.sample(find((predict_label - labels.') ~=0),:);
    plot3(class_0(:,1),class_0(:,2),class_0(:,3),'+',class_1(:,1),class_1(:,2),class_1(:,3),'*',support_vector_machines(:,1),support_vector_machines(:,2),support_vector_machines(:,3),'o')
    title(plot_title)
    legend('class_0','class_1','svm');
    saveas(gcf,plot_title,'png');
end
end

