
function is_func_def(f)
    if isa(f, Expr) && (f.head === :function || Base.is_short_function_def(f))
        return true
    else
        return false
    end
end

global EXPR = 0
macro llvmcall(expr)
    global EXPR = expr
    @assert is_func_def(expr)

    wheres = expr.args[1].args[2:end]
    decl_args = expr.args[1].args[1]
    call_args = decl_args.args[1].args
    return_type = decl_args.args[2]

    func_name = call_args[1]

    arg_names = [call_args[i].args[1] for i in 2:length(call_args)]
    arg_types = [call_args[i].args[2] for i in 2:length(call_args)]

    @assert length(expr.args) == 2
    body = expr.args[2]
    g = gensym()
    str = Expr(:block, esc.(body.args)...)
    llvm_expr = :(Base.llvmcall($(Expr(:$, :str)),
                    $return_type,
                    Tuple{$((arg_types)...)},
                    $((arg_names)...)))

    ex = :(function $(esc(func_name))($(esc.(call_args[2:end])...)) where {$(esc.(wheres)...)}
            $(esc(:str)) = $str
            return $(Expr(:quote, llvm_expr))
    end)
    return Expr(:macrocall, Symbol("@generated"), nothing, ex)
end
