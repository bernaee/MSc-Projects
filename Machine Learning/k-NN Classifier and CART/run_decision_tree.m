function  run_decision_tree(training,validation, thresholds)
for i=1:length(thresholds)
    decision_tree(training,validation, thresholds(i));
end
end

