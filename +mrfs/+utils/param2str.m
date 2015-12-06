function str = param2str( param )
%PARAM2STR Convert single param or cell array of params to string
if(iscell(param))
    str = '{';
    for i = 1:length(param)
        if(i == 1)
            str = sprintf(' %s', str, oneparam(param{i}) );
        else
            str = sprintf('%s, %s', str, oneparam(param{i}) );
        end
    end
    str = sprintf('%s }', str);
else
    str = oneparam(param);
end

    function str = oneparam( param )
        if(isnumeric( param ))
            if(numel(param)==1)
                if(round(param) == param)
                    str = sprintf('%d', round(param));
                else
                    str = sprintf('%g', param);
                end
            else
                str = mat2str([1,2,3]);
            end
        elseif(ischar( param ))
            str = param;
        elseif(isa( param, 'function_handle') )
            str = func2str(param);
        elseif(isa( param, 'mrfs.VerboseHandler'))
            str = param.name();
        else
            str = sprintf('? %s ?', class( param ));
        end
    end

end

