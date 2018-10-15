function [mean var] = calc_class_params(class)
class_size = length(class);
sum=0;
for i=1:length(class)
    sum=sum+class(i);
end
mean = sum/class_size;
sum=0;
for i=1:length(class)
    sum=sum+ (class(i)-mean)^2;
end
var=sum/length(class);
end

