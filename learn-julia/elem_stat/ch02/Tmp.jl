
module Tmp
export read_data_test, read_data_train

DATA_PATH = "./21_zip_code"
DATA_PATH_TRAIN = joinpath(DATA_PATH, "zip.train")
DATA_PATH_TEST = joinpath(DATA_PATH, "zip.test")

read_data_train() = read_data(DATA_PATH_TRAIN)
read_data_test() = read_data(DATA_PATH_TEST)

function read_data(path::String)
    raw = read_raw_data(path)
    (x, y) = process_raw_data(raw)
    return (x, y)
end


function read_raw_data(path::String)
    f = open(path, "r")
    raw = read(f, String)
    close(f)
    return raw
end


function process_raw_data(raw::String)
    lines = split(rstrip(raw), "\n")

    d = length(split(rstrip(lines[1]), " ")) - 1
    x = Array{Float64}(undef, length(lines), d)
    y = Array{Int64}(undef, length(lines))

    for (i, line) in enumerate(lines)
        if length(line) == 0
            continue
        end

        label, data = split(line, " ", limit = 2)
        y[i] = Int64(floor(parse(Float64, label)))

        image = [parse(Float64, j) for j in split(rstrip(data), " ")]
        x[i, :] = image
    end

    return (x, y)
end

struct KnnConfig
    x::Array{Float64}
    y::Array{Int64}
    k::Int64
end

# TODO K-nn algo
function knn(x_test, config)
    n = size(x_test, 1)
    y_pred = Array{Int64}(undef, n)

    for i = 1:n
        x = x_test[i, :]
        dists = sum(((config.x .- x') .^ 2), dims=2)

        idxs = sortperm(dists, dims=1)
        y_train_sort = config.y[idxs]
        y_train_sort_k = y_train_sort[1:config.k]

        votes = vcat([[i, count(==(i), y_train_sort_k)] for i in unique(y_train_sort_k)]'...)
        idx_max = argmax(votes[:, 2])
        y_pred[i] = votes[idx_max, 1]
    end

    return y_pred
end

# TODO linreg algo
# TODO predict and evaluate
function evaluate(config::KnnConfig)
    x_test, y_test = read_data_test()
    y_pred = knn(x_test, config)
    acc = sum(y_pred .== y_test) / length(y_test)
    println("Accuracy : ", acc)
end

function evaluate_knn(k::Int64)
    x_train, y_train = read_data_train()
    config = KnnConfig(x_train, y_train, k)
    evaluate(config)
end

end
