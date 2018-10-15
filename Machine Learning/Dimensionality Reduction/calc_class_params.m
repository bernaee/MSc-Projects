function [mu cov] = calc_class_params(class)
[sample_size,dim] =size(class);
mu = mean(class);
cov =((class-mu)'*(class-mu))/sample_size;
end