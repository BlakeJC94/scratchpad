include("Tmp.jl")
using Test
import .Tmp

@testset "read_raw_data" begin
    @test Tmp.read_raw_data(Tmp.DATA_PATH_TEST)[1:20] == "9 -1 -1 -1 -1 -1 -0."
    @test Tmp.read_raw_data(Tmp.DATA_PATH_TRAIN)[1:20] == "6.0000 -1.0000 -1.00"
end

@testset "process_raw_data" begin
    raw_data_test = "9 -1 -1 -0.948 -0.561\n6 -1 -0.783 -0.973 -1\n"
    x_test, y_test = Tmp.process_raw_data(raw_data_test)
    @test size(x_test) == (2, 4)
    @test size(y_test) == (2,)
    @test y_test == [9, 6]
    @test x_test == [-1 -1 -0.948 -0.561; -1 -0.783 -0.973 -1]

    raw_data_train = "6.0000 -1.0000 -1.0000 -1.0000 -1.0000 -1.0000\n5.0000 -1.0000 -1.0000 -1.0000 -0.8130 -0.6710\n"
    x_train, y_train = Tmp.process_raw_data(raw_data_train)
    @test size(x_train) == (2, 5)
    @test size(y_train) == (2,)
end

# @testset "read_data" begin
#     x_train, y_train = Tmp.read_data_train()
#     @test size(x_train) == (7291, 256)
#     @test size(y_train) == (7291,)

#     x_test, y_test = Tmp.read_data_test()
#     @test size(x_test) == (2007, 256)
#     @test size(y_test) == (2007,)
# end

@testset "KnnConfig" begin
    x_test, y_test = Tmp.read_data_test()
    k = rand(1:20)
    config = Tmp.KnnConfig(x_test, y_test, k)
    @test config.x == x_test
    @test config.y == y_test
    @test config.k == k
end

Tmp.evaluate_knn(10)

