function kNN(distance, training ,validation, k, plot_title)
S = cov(cat(1,training.sample,validation.sample));
for i=1:length(validation.sample)
    for j=1:length(training.sample)
         distances(i,j) = kernel_estimator(distance,training.sample(j,:)-validation.sample(i,:),S);
    end
end

[sorted,indexes] = sort(distances,2);
for i=1:length(validation.sample)
    votes_0 = length(find(training.label(indexes(i,1:k))==0));
    votes_1 = length(find(training.label(indexes(i,1:k))==1));
    if votes_0 > votes_1
        prediction(i,1) = 0;
    elseif votes_1 > votes_0
        prediction(i,1) = 1;
    end
end

tp = length(find(validation.label(find(validation.label==prediction))==0));
fn =  length(find(validation.label(find(validation.label~=prediction))==0));
fp =  length(find(validation.label(find(validation.label~=prediction))==1));
tn =  length(find(validation.label(find(validation.label==prediction))==1));
cm=[[tp,fn];[fp,tn]]
class_0= validation.sample(prediction == 0,:);
class_1= validation.sample(prediction == 1,:);
plot3(class_0(:,1),class_0(:,2),class_0(:,3),'+',class_1(:,1),class_1(:,2),class_1(:,3),'*');
title(plot_title)
legend('class_0','class_1');
saveas(gcf,plot_title,'png');
end

