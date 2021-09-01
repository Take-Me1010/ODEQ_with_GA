
# ref: https://qiita.com/AokiMasataka/items/4fe9b2623282238ba6f3

using Random

Random.seed!(64)

# 集団を表す型エイリアス。Int8の二次元配列の意味。各個体の遺伝子は行で表される。
Population = Matrix{Int8}

# 個体を表すエイリアス
Individual = Vector{Int8}

# 個体の評価値をまとめた行列。 n_population × 1 Matrix
EvaluatedIndividuals = Matrix{Int8}

"""
    init(genome_length, n_population)

初期集団を生成して返す。
"""
function init(genome_length::Int, n_population::Int):: Population
    population::Population = rand(0:1, n_population, genome_length)
    return population
end

"""
    evaluate(population)

各個体の評価値を返す。n_population × 1 Matrixを返す。
"""
function evaluate(population::Population):: EvaluatedIndividuals
    return sum(population, dims=2)
end

"""
    choice(population)

最良個体と、ランダムに選ばれた個体を選んで返す。
"""
function choice(population::Population):: Tuple{Individual, Individual}
    bestIndex = argmax(evaluate(population))[1]
    # 最良個体
    x::Individual = population[bestIndex, :]
    # ランダム個体。bestIndexを除いたindexからランダムに選ぶ。
    n_population = size(population)[1]
    y::Individual = population[rand(setdiff(1:n_population, [bestIndex])), :]
    return x, y
end

"""
    crossover(x, y)

交差した個体を返す。一点交叉。
"""
function crossover(x::Individual, y::Individual):: Individual
    # 遺伝子の長さ
    length = size(x)[1]

    # 切り取るindex
    cut = rand(2: length-1)
    return cat(x[1: cut], y[cut+1: length]; dims=1)
end

"""
    mutate!(population, rate)

突然変異を起こす。各遺伝子に確率rateで反転が起きる。
"""
function mutate!(population::Population, rate:: Number) ::Population
    for i in eachindex(population)
        if rand() < rate
            population[i] = 1 - population[i]
        end
    end
    return population
end

"""
    next_generation!(population [, rate])

次世代を生成して返却する。rateは突然変異の確率。
"""
function next_generation!(population::Population; rate::Number = 0.01):: Population
    x, y = choice(population)
    population[1, :] = x'
    population[2, :] = y'

    n_population = size(population)[1]

    for i in 3:n_population
        offspring = crossover(x, y)
        population[i, :] = offspring'
    end

    population = mutate!(population, rate)

    return population
end


function main()
    length = 50
    n_population = 100
    mutationRate = 0.01
    generation = 50

    best_result = length

    population:: Population = init(length, n_population)

    for i in 1: generation
        next_generation!(population; rate=mutationRate)

        evaluated = evaluate(population)

        println("Generation $(i) ------------")
        print("avg: ")
        println(sum(evaluated) / n_population / best_result)
        print("max: ")
        println(maximum(evaluated) / best_result)
        println("")
    end
end

main()
