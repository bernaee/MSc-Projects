function decision_tree(training,validation, threshold)
tree= build_tree(training, threshold);
[dim,sample]=size(validation.x);
for i=1:sample
    prediction(i)=predict_tree(tree,validation.x(i));
end
 
tr_error = tree.label;
val_error = sum((validation.r - prediction).^2)/sample;

plot_title = sprintf('The Complexity=%.2f',threshold);
fprintf(plot_title)
fprintf('\n')
fprintf(' Training error: %.2f \n',tr_error)
fprintf(' Validation error: %.2f \n',val_error)
plot(validation.x,validation.r,'.',validation.x,prediction);
title(plot_title);
legend('val','pred');
saveas(gcf,strcat(plot_title,'.png'),'png'); 
traverse(tree);
  
end

