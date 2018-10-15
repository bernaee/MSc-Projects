function idx = find_decision_boundary(class, class_1,class_2)
A = abs(class_1-class_2);
idx = class(find(A == min(A)));
end

