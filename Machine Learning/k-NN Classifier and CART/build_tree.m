function [node] = build_tree(training, threshold)
[dim,sample]=size(training.x);
if sample <=2
    node=mean(training.r);
else 
    node.label = mean(training.r);
    imp_node = sum((training.r-node.label).^2)/length(training.r);
    max_imp=99999;
    if imp_node > threshold
        for i=1:sample        
            [left,right] = split(training.x(i), training);
            imp_left = sum((left.r-mean(left.r)).^2);
            imp_right = sum((right.r-mean(right.r)).^2);
            imp_branch = (imp_left + imp_right)/(length(left.x)+length(right.x));
            if (imp_branch) < max_imp
                max_imp = (imp_branch);
                best_split = training.x(i);
            end
        end
        node.split = best_split;
        [node_left, node_right] = split(best_split, training);
        node.left = build_tree(node_left, threshold );
        node.right = build_tree(node_right, threshold );         
    else
       node=mean(training.r);
    end
end
end
