function [th_1, th_2] = find_decision_boundary(class, class_1,class_2)
A = abs(class_1-class_2);
sorted_A = sort(A);
th_1 = class(find(A == min(A)));
th_2 = [];
for i = 1:numel(sorted_A)
    temp_min = class(find(A == sorted_A(i)));
    if round(abs(th_1 -temp_min),0)>0.0
        th_2 = temp_min;
        break
    end
end
end

