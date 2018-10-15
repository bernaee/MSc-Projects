function varargout = split_train_validation_data(data)
sample = data(:,1:3);
label= data(:,4);
class_0 = sample(label == 0,:);
class_1 = sample(label == 1,:);
label_0=label == 0;
label_1=label == 1;
training.sample = cat(1,class_0(1:60,:), class_1(1:60,:)) ;
training.label = cat(1,label_0(1:60,:), label_1(1:60,:)) ;
validation.sample =  cat(1,class_0(61:100,:), class_1(61:100,:));
validation.label = cat(1,label_0(61:100,:), label_1(61:100,:));
varargout = {training, validation};
end

