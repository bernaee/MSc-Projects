function display_tree(node, depth)
if isstruct(node)  
    fprintf('%s x < %d \n',repmat('-',1,depth),node.split)
    display_tree(node.left, depth + 1)
    display_tree(node.right, depth + 1)
end
end

