function varargout = split_train_validation_data(sample,label)
class_0 = sample(label == 0,:);
class_1 = sample(label == 1,:);
training = cat(1,class_0(1:60,:), class_1(1:60,:)) ;
training_labels = cat(1,label(1:60,:), label(101:160,:)) ;
validation =  cat(1,class_0(61:100,:), class_1(61:100,:));
validation_labels = cat(1,label(61:100,:), label(161:200,:));
training_0 = class_0(1:60,:);
training_1 = class_1(1:60,:);
validation_0 = class_0(61:100,:);
validation_1=class_1(61:100,:);
varargout = {training_labels,class_0,class_1,training_0,training_1,validation_0,validation_1};
end

