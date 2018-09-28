using JuMP, Gurobi

type Mkpdata
    # Data for mkp instances
    name
    num_items
    num_knapsacks
    value
    weight
    capacity
end
Mkpdata() = Mkpdata("", 0, 0, 0, 0, 0)
# constructor

function readdatafromfile(datafilename)
    # return Mkpdata given data path
    datafile = open(datafilename)
    data = readlines(datafile)
    close(datafile)
    filedata = Int[]
    for line in data
        append!(filedata, [parse(Int, s) for s in split(line)])
    end
    instancedata = Mkpdata()
    instancedata.name = datafilename
    instancedata.num_items = filedata[1]
    instancedata.num_knapsacks = filedata[2]
    instancedata.value = filedata[4:3+instancedata.num_items]
    instancedata.capacity = filedata[end-instancedata.num_knapsacks+1:end]
    weightdata = filedata[4+instancedata.num_items:end-instancedata.num_knapsacks]
    instancedata.weight = transpose(reshape(weightdata, instancedata.num_items, instancedata.num_knapsacks))
    return(instancedata)
end

function solve_mkp(datafilename)
    # Solve Mkp instances using MIP
    if typeof(datafilename) == String
        d = readdatafromfile(datafilename)
    else
        d = datafilename
    end
    m = Model(solver=GurobiSolver(Heuristics=0, Presolve=0, Cuts=0, OutputFlag = 0))
    @variable(m, x[1:d.num_items], Bin)
    @objective(m, Max, dot(x, d.value))
    @constraint(m, constraint[j=1:d.num_knapsacks], dot(d.weight[j,:], x) <= d.capacity[j])
    solve(m)
    return m.objVal, getvalue(x)
end

function solve_lr(datafilename)
    # Solve the Linear Relaxation of Mkp instances
    if typeof(datafilename) == String
        d = readdatafromfile(datafilename)
    else
        d = datafilename
    end
    m = Model(solver=GurobiSolver(Heuristics=0, Presolve=0, Cuts=0, OutputFlag = 0))
    @variable(m, 0<=x[1:d.num_items]<=1)
    @objective(m, Max, dot(x, d.value))
    @constraint(m, constraint[j=1:d.num_knapsacks], dot(d.weight[j,:], x) <= d.capacity[j])
    solve(m)
    return m.objVal, getvalue(x)
end

function primalheuristic(datafilename; passes=100)
    if typeof(datafilename) == String
        d = readdatafromfile(datafilename)
    else
        d = datafilename
    end
    preorder(d)
    cap = d.capacity
    vallr, sollr = solve_lr(d)
    solheu = zeros(Int, d.num_items)
    solfixed = zeros(Int, d.num_items)
    objheu = 0
    for i in 1 : d.num_items
        xi = sollr[i]
        if xi == 1
            solfixed[i] = 1
            cap -= d.weight[:,i]
        end
    end
    for i in 1:passes
        solpass = solfixed
        for j in 1:d.num_items
            if minimum(cap - d.weight[:,j]) < 0
                continue
            end
            xj = sollr[j]
            if xj != round(xj)
                rd = rand()
                if xj >= rd
                    solpass[j] = 1
                    cap -= d.weight[:,j]
                end
            end
        end
        objpass = dot(solpass, d.value)
        if objpass > objheu
            objheu = objpass
            solheu = solpass
        end
    end
    return objheu
end

function preorder(d::Mkpdata)
    value = deepcopy(d.value)
    weight = deepcopy(d.weight)
    score = zeros(Float64, d.num_items)
    for j in 1:d.num_items
        score[j] = value[j]/sqrt(sum(weight[:,j]./d.capacity))
    end
    orderedindex = sortperm(score,rev=true)
    for i in 1:d.num_items
        d.value[i] = value[orderedindex[i]]
        d.weight[:,i] = weight[:,orderedindex[i]]
    end
end

type Node
    # First (item_index - 1) items are fixed, with corresponding profit, weight and upper bound
    item_index
    profit
    upper
    weight
end
Node() = Node(0, 0, 0, 0)




function upper_bound(u::Node, d::Mkpdata)
    if minimum(d.capacity - u.weight) < 0
        return 0
    end
    j = u.item_index
    max_avg_price = maximum([d.value[i] / sum(d.weight[:,i]) for i in j:d.num_items])
    remain_cap = d.capacity - u.weight
    profit_upperbound = max_avg_price * sum(remain_cap) + u.profit
    return profit_upperbound
end

function lp_bound(u::Node, d::Mkpdata)
    if minimum(d.capacity - u.weight) < 0
        return 0
    end
    j = u.item_index + 1
    value = deepcopy(d.value)
    weight = deepcopy(d.weight)
    capacity = deepcopy(d.capacity)
    sub_d = Mkpdata()
    sub_d.num_items = d.num_items - j + 1
    sub_d.num_knapsacks = d.num_knapsacks
    sub_d.value = value[j:d.num_items]
    sub_d.weight = weight[:,j:d.num_items]
    sub_d.capacity = capacity - u.weight
    profit_upperbound = u.profit + solve_lr(sub_d)[1]
    return profit_upperbound
end

# function surrogate_bound(u::Node, d::Mkpdata)
#     if minimum(d.capacity - u.weight) < 0
#         return 0
#     end
#     j = u.item_index + 1
#     value = d.value[j:d.num_items]
#     weight = d.weight[:,j:d.num_items]
#     capacity = d.capacity - u.weight
#     num_items = d.num_items - j + 1
#     m = Model(solver=GurobiSolver(Heuristics=0, Presolve=0, Cuts=0, OutputFlag = 0))
#     @variable(m, 0<=mu[1:d.num_knapsacks]<=1)
#     function objfun(y)
#         surrogate_weight = vec(vec(y).' * weight)
#         surrogate_cap = dot(y, capacity)
#         ratio = surrogate_weight ./ value
#         return 0
#     end
#     @objective(m, Min, objfun(mu))
#     solve(m)
#     mu = getvalue(mu)
#     mm = Model(solver=GurobiSolver(Heuristics=0, Presolve=0, Cuts=0, OutputFlag = 0))
#     @variable(mm, x[1:num_items], Bin)
#     @objective(mm, Max, dot(x, value))
#     @constraint(mm, dot(  vec(  vec(mu).' * weight ), x )<= dot(mu, capacity)  )
#     solve(mm)
#     profit_upperbound = mm.objVal + u.profit
#     return profit_upperbound
# end

function camus(d::Mkpdata)
    # Leyton-Brown et al. 2003
    preorder(d)
    Q = []
    u = Node(0, 0, 0, zeros(Int, d.num_knapsacks))
    push!(Q, u)
    max_profit = primalheuristic(d,passes=1000)
    n = 1
    while !isempty(Q)
        n += 1
        parent = shift!(Q)
        child_1 = Node()
        child_1.item_index = parent.item_index + 1
        # println(child_1.item_index)
        child_1.weight = parent.weight + d.weight[:,child_1.item_index]
        child_1.profit = parent.profit + d.value[child_1.item_index]
        if (minimum(d.capacity - child_1.weight) >= 0) & (child_1.profit > max_profit)
            max_profit = child_1.profit
        end
        child_1.upper = upper_bound(child_1, d)
        if (child_1.upper > max_profit) & (child_1.item_index < d.num_items)
            push!(Q, child_1)
        end
        child_0 = Node()
        child_0.item_index = parent.item_index + 1
        child_0.weight = parent.weight
        child_0.profit = parent.profit
        child_0.upper = upper_bound(child_0, d)
        if (child_0.upper > max_profit) & (child_0.item_index < d.num_items)
            push!(Q, child_0)
        end
    end
    # println(n)
    # println(max_profit)
    return n
end

function branch_and_cut(d::Mkpdata)
    # Leyton-Brown et al. 2003
    preorder(d)
    Q = []
    u = Node(0, 0, 0, zeros(Int, d.num_knapsacks))
    push!(Q, u)
    max_profit = primalheuristic(d)
    n = 1
    while !isempty(Q)
        n += 1
        parent = shift!(Q)
        child_1 = Node()
        child_1.item_index = parent.item_index + 1
        # println(child_1.item_index)
        child_1.weight = parent.weight + d.weight[:,child_1.item_index]
        child_1.profit = parent.profit + d.value[child_1.item_index]
        if (minimum(d.capacity - child_1.weight) >= 0) & (child_1.profit > max_profit)
            max_profit = child_1.profit
        end
        child_1.upper = cover_bound(child_1, d)
        if (child_1.upper > max_profit) & (child_1.item_index < d.num_items)
            push!(Q, child_1)
        end
        child_0 = Node()
        child_0.item_index = parent.item_index + 1
        child_0.weight = parent.weight
        child_0.profit = parent.profit
        child_0.upper = cover_bound(child_0, d)
        if (child_0.upper > max_profit) & (child_0.item_index < d.num_items)
            push!(Q, child_0)
        end
    end
    # println(n)
    # println(max_profit)
    return n
end

function branch_and_bound(d::Mkpdata)
    # Leyton-Brown et al. 2003
    preorder(d)
    Q = []
    u = Node(0, 0, 0, zeros(Int, d.num_knapsacks))
    push!(Q, u)
    max_profit = primalheuristic(d)
    n = 1
    while !isempty(Q)
        n += 1
        parent = shift!(Q)
        child_1 = Node()
        child_1.item_index = parent.item_index + 1
        # println(child_1.item_index)
        child_1.weight = parent.weight + d.weight[:,child_1.item_index]
        child_1.profit = parent.profit + d.value[child_1.item_index]
        if (minimum(d.capacity - child_1.weight) >= 0) & (child_1.profit > max_profit)
            max_profit = child_1.profit
        end
        child_1.upper = lp_bound(child_1, d)
        if (child_1.upper > max_profit) & (child_1.item_index < d.num_items)
            push!(Q, child_1)
        end
        child_0 = Node()
        child_0.item_index = parent.item_index + 1
        child_0.weight = parent.weight
        child_0.profit = parent.profit
        child_0.upper = lp_bound(child_0, d)
        if (child_0.upper > max_profit) & (child_0.item_index < d.num_items)
            push!(Q, child_0)
        end
    end
    # println(n)
    # println(max_profit)
    return n
end

function reduction(d::Mkpdata)
    n = 0
    lower_bound = solve_mkp(d)[1]
    for i in 1:d.num_items
        m = Model(solver=GurobiSolver(Heuristics=0, Presolve=0, Cuts=0, OutputFlag = 0))
        @variable(m, x[1:d.num_items], Bin)
        @objective(m, Max, dot(x, d.value))
        @constraint(m, constraint[j=1:d.num_knapsacks], dot(d.weight[j,:], x) <= d.capacity[j])
        @constraint(m, temp, x[i] == 0)
        solve(m)
        if m.objVal <= lower_bound + 1
            n += 1
        end
    end
    for i in 1:d.num_items
        m = Model(solver=GurobiSolver(Heuristics=0, Presolve=0, Cuts=0, OutputFlag = 0))
        @variable(m, x[1:d.num_items], Bin)
        @objective(m, Max, dot(x, d.value))
        @constraint(m, constraint[j=1:d.num_knapsacks], dot(d.weight[j,:], x) <= d.capacity[j])
        @constraint(m, temp, x[i] == 1)
        solve(m)
        if m.objVal <= lower_bound + 1
            n += 1
        end
    end
    return n
end

function primalheuristic2(datafilename; passes=10)
    if typeof(datafilename) == String
        d = readdatafromfile(datafilename)
    else
        d = datafilename
    end
    preorder(d)
    cap = d.capacity
    vallr, sollr = solve_lr(d)
    solheu = zeros(Int, d.num_items)
    objheu = 0
    for i in 1:passes
        solpass = zeros(Int, d.num_items)
        for j in 1:d.num_items
            if minimum(cap - d.weight[:,j]) < 0
                continue
            end
            xj = sollr[j]
            rd = rand()
            if xj >= rd
                solpass[j] = 1
                cap -= d.weight[:,j]
            end
        end
        objpass = dot(solpass, d.value)
        if objpass > objheu
            objheu = objpass
            solheu = solpass
        end
    end
    return objheu
end

function glc_upper_bound(u::Node, d::Mkpdata)
    if minimum(d.capacity - u.weight) < 0
        return 0
    end
    j = u.item_index + 1
    value = deepcopy(d.value)
    weight = deepcopy(d.weight)
    capacity = deepcopy(d.capacity)
    sub_d = Mkpdata()
    sub_d.num_items = d.num_items - j + 1
    sub_d.num_knapsacks = d.num_knapsacks
    sub_d.value = value[j:d.num_items]
    sub_d.weight = weight[:,j:d.num_items]
    sub_d.capacity = capacity - u.weight
    m = Model(solver=GurobiSolver(Heuristics=0, Presolve=0, Cuts=0, OutputFlag = 0))
    @variable(m, 0<=x[1:sub_d.num_items]<=1)
    @objective(m, Max, dot(x, sub_d.value))
    @constraint(m, constraint[j=1:sub_d.num_knapsacks], dot(sub_d.weight[j,:], x) <= sub_d.capacity[j])
    solve(m)
    x = getvalue(x)
    println(x)
    s = sortperm(x,rev=true)
    for i in 1:sub_d.num_knapsacks
        thisweight = sub_d.weight[i,:]
        thiscap = sub_d.capacity[i]
        sum_a = 0
        Cover = []
        for k in s
            sum_a += thisweight[k]
            push!(Cover, k)
            if sum_a > thiscap
                break
            end
        end
        L1 = []
        for k in reverse(Cover)
                if (sum_a - thisweight[k]) > thiscap
                    sum_a -= thisweight[k]
                    push!(L1, k)
                end
        end
        Cover = setdiff(Cover,L1)
        D = find(x->abs(x-1)<10e-8, x)
        D = intersect(D, Cover)
        L0 = find(x->abs(x)<10e-8, x)
        Lf = find(x->(abs(x)>10e-8)&(abs(x-1)>10e-8), x)
        Lf = setdiff(Lf, Cover)
        Lf = setdiff(Lf, L1)
        println("------------------------------------------------")
        print("Cover: ")
        println(Cover)
        print("D: ")
        println(D)
        print("L1: ")
        println(L1)
        print("Lf: ")
        println(Lf)
        print("L0: ")
        println(L0)
        println(length(L1)+length(Lf)+length(D)+length(L0))
    end
end

function heuristic1_bound(u::Node, d::Mkpdata)
    if minimum(d.capacity - u.weight) < 0
        return 0
    end
    j = u.item_index + 1
    value = deepcopy(d.value)
    weight = deepcopy(d.weight)
    capacity = deepcopy(d.capacity)
    sub_d = Mkpdata()
    sub_d.num_items = d.num_items - j + 1
    sub_d.num_knapsacks = d.num_knapsacks
    sub_d.value = value[j:d.num_items]
    sub_d.weight = weight[:,j:d.num_items]
    sub_d.capacity = capacity - u.weight
    bounds = []
    for j in 1:sub_d.num_knapsacks
        this_m = Model(solver=GurobiSolver(Heuristics=0, Presolve=0, Cuts=0, OutputFlag = 0))
        @variable(this_m, 0<=x[1:sub_d.num_items]<=1)
        @objective(this_m, Max, dot(x, sub_d.value))
        @constraint(this_m, constraint, dot(sub_d.weight[j,:], x) <= sub_d.capacity[j])
        solve(this_m)
        push!(bounds, this_m.objVal)
    end
    profit_upperbound = u.profit + minimum(bounds)
    return profit_upperbound
end

function cover_bound(u::Node, d::Mkpdata)
    if minimum(d.capacity - u.weight) <= 0
        return 0
    end
    j = u.item_index + 1
    num_items = d.num_items - j + 1
    value = d.value[j:d.num_items]
    weight = d.weight[:,j:d.num_items]
    capacity = d.capacity - u.weight
    m = Model(solver=GurobiSolver(Heuristics=0, Presolve=0, Cuts=0, OutputFlag = 0))
    @variable(m, 0<=x[1:num_items]<=1)
    @objective(m, Max, dot(x, value))
    @constraint(m, constraint[i=1:d.num_knapsacks], dot(weight[i,:], x) <= capacity[i])
    solve(m)
    x_val = getvalue(x)
    s = sortperm(x_val, rev=true)
    for i in 1:d.num_knapsacks
        thisweight = weight[i,:]
        thiscap = capacity[i]
        sum_a = 0
        Cover = []
        for k in s
            sum_a += thisweight[k]
            push!(Cover, k)
            if sum_a > thiscap
                break
            end
        end
        L1 = []
        for k in reverse(Cover)
            if (sum_a - thisweight[k]) >= thiscap
                sum_a -= thisweight[k]
                push!(L1, k)
            end
        end
        Cover = setdiff(Cover, L1)
        con = @constraint(m, sum(x[Cover]) <= length(Cover) - 1)
    end
    solve(m)
    return m.objVal + u.profit
end

function bb(d::Mkpdata)
    preorder(d)
    m = Model(solver=GurobiSolver(Heuristics=0, Presolve=1, Cuts=0, OutputFlag = 1))
    @variable(m, 0<=x[1:d.num_items]<=1, Int)
    @objective(m, Max, dot(x, d.value))
    @constraint(m, constraint[j=1:d.num_knapsacks], dot(d.weight[j,:], x) <= d.capacity[j])

    function myheuristic(cb)
        cap = d.capacity
        x_val = getvalue(x)
        objheu = 0
        for i in 1:d.num_items
            xi = x_val[i]
            if xi == 1
                setsolutionvalue(cb, x[j], 1)
                addsolution(cb)
                cap -= d.weight[:,i]
            end
        end
        for j in 1:d.num_items
            if minimum(cap - d.weight[:,j]) < 0
                continue
            end
            xj = x_val[j]
            if xj != round(xj)
                rd = rand()
                if xj >= rd
                    setsolutionvalue(cb, x[j], 1)
                    addsolution(cb)
                    cap -= d.weight[:,j]
                end
            end
        end
    end
    function mycutgenerator(cb)
        x_val = getvalue(x)
        s = sortperm(x_val, rev=true)
        for i in 1:d.num_knapsacks
            thisweight = d.weight[i,:]
            thiscap = d.capacity[i]
            sum_a = 0
            Cover = []
            for k in s
                sum_a += thisweight[k]
                push!(Cover, k)
                if sum_a > thiscap
                    break
                end
            end
            L1 = []
            for k in reverse(Cover)
                if (sum_a - thisweight[k]) > thiscap
                    sum_a -= thisweight[k]
                    push!(L1, k)
                end
            end
            Cover = setdiff(Cover, L1)
            @usercut(cb, sum(x[Cover]) <= length(Cover) - 1)
        end
    end
    addcutcallback(m, mycutgenerator)
    # addheuristiccallback(m, myheuristic)
    solve(m)
end

function generate_instance(num_items, num_knapsacks)
    d = Mkpdata()
    d.num_items = num_items
    d.num_knapsacks = num_knapsacks
    d.value = rand(1:100, num_items)
    d.weight = rand(1:100, (num_knapsacks, num_items))
    d.capacity = round(sum(d.weight, 2)/2)
    return d
end

function compare_methods(num_items, num_knapsacks, num_instances)
    avg_num_nodes_lehmann = 0
    avg_time_lehmann = 0
    avg_num_nodes_lp = 0
    avg_time_lp = 0
    avg_num_nodes_cut = 0
    avg_time_cut = 0
    for i in 1:num_instances
        # print(i)
        # print("/")
        # print(num_instances)
        # println(" ")
        d = generate_instance(num_items, num_knapsacks)
        n1, t1 = @timed camus(d)
        n2, t2 = @timed branch_and_bound(d)
        n3, t3 = @timed branch_and_cut(d)
        avg_num_nodes_lehmann += n1
        avg_time_lehmann += t1
        avg_num_nodes_lp += n2
        avg_time_lp += t2
        avg_num_nodes_cut += n3
        avg_time_cut += t3
    end
    avg_num_nodes_lehmann /= num_instances
    avg_time_lehmann /= num_instances
    avg_num_nodes_lp /= num_instances
    avg_time_lp /= num_instances
    avg_num_nodes_cut /= num_instances
    avg_time_cut /= num_instances
    return [avg_num_nodes_lehmann, avg_time_lehmann, avg_num_nodes_lp, avg_time_lp, avg_num_nodes_cut, avg_time_cut]
end

compare_methods(10,10,10) # warmup


function autorun()
    println(compare_methods(10,5,100))
    println(compare_methods(20,10,100))
    # println(compare_methods(30,20,100))
    # println(compare_methods(40,30,100))
    # println(compare_methods(50,40,100))
end

autorun()
