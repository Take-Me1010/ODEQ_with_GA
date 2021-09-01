
# ref: https://qiita.com/AokiMasataka/items/4fe9b2623282238ba6f3

using Random

Random.seed!(0)

# 集団を表す型エイリアス。Int8の二次元配列の意味。各個体の遺伝子は行で表される。
Population = Matrix{Int8}

# 個体を表すエイリアス
Individual = Vector{Int8}

# 個体の評価値をまとめた行列。 n_population × 1 Matrix
EvaluatedIndividuals = Matrix{Int}

"""
    init(genome_length, n_population)

初期集団を生成して返す。
"""
function init(genome_length::Int, n_population::Int):: Population
    population::Population = rand(0:1, n_population, genome_length)
    return population
end

"""
    bin2den(population)

populationを各個体が表す10進数の値の行列にして返す。
"""
function bin2den(population::Population):: EvaluatedIndividuals
    length = size(population)[2]
    # [2^(l-1), ... , 2^1, 2^0]
    binDigit = 2 .^ (length-1:-1:0)'
    return sum(population .* binDigit, dims=2)
end

"""
    evaluate(population)

各個体の評価値を返す。n_population × 1 Matrixを返す。
"""
function evaluate(population::Population):: EvaluatedIndividuals
    return abs.(bin2den(population) .- 5)
end

"""
    select(population, k)

評価の高い個体をk個体選んで返す。
"""
function select(population::Population, k::Int):: Population
    n_population = size(population)[1]
    evaluated:: EvaluatedIndividuals = evaluate(population)
    sorted = population[sortperm(evaluated[:]), :]
    selected = sorted[1:k, :]

    return selected
end

"""
    crossover(x, y)

両親x, yから2個体を新しく作り出して返却する。一様交叉。
"""
function crossover(x::Individual, y::Individual):: Tuple{Individual, Individual}
    boy = similar(x)
    girl = similar(y)

    for i in eachindex(x)
        if rand() >= 0.5
            boy[i] = x[i]
            girl[i] = y[i]
        else
            boy[i] = y[i]
            girl[i] = x[i]
        end
    end

    return boy, girl
end

"""
    mutate!(population, rate)

突然変異を起こす。各遺伝子に確率rateで反転が起きる。
"""
function mutate!(population::Population, μ:: Number) ::Population
    for i in eachindex(population)
        if rand() < μ
            population[i] = 1 - population[i]
        end
    end
    return population
end

"""
    next_generation!(population [, μ])

次世代を生成して返却する。μは突然変異の確率。

# Attention

n_populationは４の倍数でなければならない。
上位N//2が生存し、その集団がN//4カップル作り、各々が2個体を増やすことで集団をN個体に維持する。
"""
function next_generation!(population::Population; μ::Number = 0.01):: Population

    @assert 0 < μ < 1 "rate must be larger than 0 and smaller than 1."

    n_population, l = size(population)
    
    @assert n_population % 4 == 0 "n_population must be shown by 4×N where N is Integer."

    # ÷ は少数切り捨て演算
    k::Int = n_population ÷ 2
    # 自然淘汰
    population = select(population, k)

    # 子孫の集団を宣言
    offsprings = similar(population, k, l)

    # カップルの数
    n_couple = k ÷ 2

    for i in 1:n_couple
        dad = population[i, :]
        mom = population[i+n_couple, :]
        boy, girl = crossover(dad, mom)
        offsprings[i, :] = boy'
        offsprings[i+n_couple, :] = girl'
    end

    # 次世代生成
    population = cat(population, offsprings, dims=1)

    # 突然変異
    population = mutate!(population, μ)

    return population
end

function main()
    genome_length = 5
    n_population = 4 * 3
    mutationRate = 0.01
    generation = 50

    population:: Population = init(genome_length, n_population)

    for i in 1:generation
        population = next_generation!(population; μ=mutationRate)

        evaluated = evaluate(population)

        println("Generation $(i) ------------")
        print("avg: ")
        println(sum(evaluated) / n_population)
        print("min: ")
        println(minimum(evaluated))
        println("")
    end

    best = select(population, 1)
    println("x = ", bin2den(best)[1])

end

main()
