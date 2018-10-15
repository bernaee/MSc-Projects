function  traverse(tree)
current = [tree];
depth = 0;
while length(current) >=1   
    depth = depth + 1;
    next = [];
    txt = '';
    for i=1:length(current)
      txt = strcat( txt,  fprintf(' x < %.2f',current(i).split)) ;  
      if isstruct(current(i).left)
         next=[next,current(i).left];
      end
      if isstruct(current(i).right)
         next=[next,current(i).right];
      end
    end
    fprintf(txt)
    fprintf('\n')
    current = next;       
end
display(depth)
end

