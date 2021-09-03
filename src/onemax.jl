
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
    select(population)

最良個体と、ランダムに選ばれた個体を選んで返す。
"""
function select(population::Population, k::Integer):: Population
    evaluated = evaluate(population)
    sorted = population[sortperm(evaluated[:]), :]
    return sorted[1:k, :]
end

"""
    crossover(dad, mom)

交差した個体を返す。2点交叉。
"""
function crossover(dad::Individual, mom::Individual):: Tuple{Individual, Individual}
    # 遺伝子の長さ
    gl = size(dad)[1]

    # 切り取るindex
    cut1 = rand(2:gl-2)
    cut2 = rand(cut1+1:gl)

    boy = similar(dad)
    girl = similar(mom)

    boy[1:cut1] = dad[1:cut1]
    girl[1:cut1] = mom[1:cut1]

    boy[cut1:cut2] = mom[cut1:cut2]
    girl[cut1:cut2] = dad[cut1:cut2]

    boy[cut2:gl] = mom[cut2:gl]
    girl[cut2:gl] = dad[cut2:gl]

    return boy, girl
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
function next_generation!(population::Population; rate = 0.01):: Population
    n_population = size(population)[1]
    k = n_population ÷ 2

    population = select(population, k)
    population = shuffle(population)

    offsprings = similar(population)
    
    n_couple = k÷2
    for i in 1:n_couple
        dad = population[i, :]
        mom = population[i+n_couple, :]
        boy, girl = crossover(dad, mom)
        offsprings[i, :] = boy'
        offsprings[i+n_couple, :] = girl'
    end

    population = cat(population, offsprings, dims=1)

    population = mutate!(population, rate)

    return population
end


function main()
    length = 10
    n_population = 12
    mutationRate = 0.01
    generation = 10

    @assert 0 < mutationRate < 1 "rate must be larger than 0 and smaller than 1."
    @assert n_population % 4 == 0 "n_population must be shown by 4×N where N is Integer."

    best_result = length

    population:: Population = init(length, n_population)

    for i in 1: generation
        population = next_generation!(population; rate=mutationRate)

        evaluated = evaluate(population)

        println("Generation $(i) ------------")
        print("avg: ")
        println(sum(evaluated) / n_population / best_result)
        print("best: ")
        println(minimum(evaluated) / best_result)
        println("")
    end
    best = select(population, 1)
    @show best
end

main()
