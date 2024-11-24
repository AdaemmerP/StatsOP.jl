using OrdinalPatterns
using Test

@testset "check frequencies" begin

    lookup_array_sop = compute_lookup_array()
    n_sops = 24
    sop = zeros(4)
    data = [9 1 2; 
            6 7 6; 
            9 4 9]
    m = size(data, 1) - 1
    n = size(data, 2) - 1
    win = zeros(Int, 4)
    
    # Check ranks    
    order_vec!([data[1, 1], data[1, 2], data[2, 1], data[2, 2]], win) 
    @test win == [2, 3, 4, 1]
    order_vec!([data[2, 1], data[2, 2], data[3, 1], data[3, 2]], win) 
    @test win == [4, 1, 2, 3]
    order_vec!([data[1, 2], data[1, 3], data[2, 2], data[2, 3]], win)  
    @test win == [1, 2, 4, 3]
    order_vec!([data[2, 2], data[2, 3], data[3, 2], data[3, 3]], win)
    @test win == [3, 2, 1, 4]

    # run sop_frequencies
    test_freq = sop_frequencies(m, n, lookup_array_sop, n_sops, data, sop)
    @test sum(findall(test_freq .== 1) .== [2, 10, 15, 19]) == 4

end

