function pred = predict_tree(tree,sample)
if sample < tree.split
        if isstruct(tree.left)
            pred = predict_tree(tree.left,sample);
        else
            pred = tree.label;
        end
else
        if isstruct(tree.right)
            pred = predict_tree(tree.right,sample);
        else
            pred = tree.label;
        end
end    
end

