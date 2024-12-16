
function test(freq_sop, seq)
  sum_1 = 0.0
  for i in seq
    sum_1 += freq_sop[i]
  end
  #freq_view = view(freq_sop, [1, 3, 8, 11, 14, 17, 22, 24])
  #sum!(r, freq_view)
end

p_hat = zeros(3)
freq_sop = rand(24)
r = [0.0]
seq = [1, 3, 8, 11, 14, 17, 22, 24]
@btime test($freq_sop, $seq)

@btime @views $freq_sop[[1, 3, 8, 11, 14, 17, 22, 24]]

p_ewma = rand(3)
@btime chart_stat_sop($p_ewma, 3)

function test2(p_ewma, p_hat, lam)
  @. p_ewma = (1 - lam) * p_ewma + lam * p_hat
end

p_ewma = rand(3)
p_hat = rand(3)
lam = 0.1
@btime test2($p_ewma, $p_hat, $lam)

function rl_sop(m, n, lookup_array_sop, lam, cl, reps_range, chart_choice, dist)

  # Pre-allocate
  n_sops = 24 # factorial(4)
  freq_sop = zeros(Int, n_sops)
  win = zeros(Int, 4)
  data_tmp = empty_data(m, n, dist)
  p_ewma = zeros(3)
  p_hat = zeros(3)
  rls = zeros(Int, length(reps_range))
  sop = zeros(4)

  # Pre-allocate for sum of frequencies
  s_1 = [1, 3, 8, 11, 14, 17, 22, 24]
  s_2 = [2, 5, 7, 9, 16, 18, 20, 23]
  s_3 = [4, 6, 10, 12, 13, 15, 19, 21]

  for r in 1:length(reps_range)
    fill!(p_ewma, 1.0 / 3.0)
    stat = chart_stat_sop(p_ewma, chart_choice)

    rl = 0

    while abs(stat) < cl
      rl += 1

      # Fill data 
      rand!(dist, data_tmp)

      # Add noise when using count data
      if dist isa DiscreteUnivariateDistribution
        for j in 1:size(data_tmp, 2)
          for i in 1:size(data_tmp, 1)
            data_tmp[i, j] = data_tmp[i, j] + rand()
          end
        end
      end

      # Compute frequencies of SOPs
      sop_frequencies!(m, n, lookup_array_sop, data_tmp, sop, win, freq_sop)

      #@views p_hat[1] = sum(freq_sop[[1, 3, 8, 11, 14, 17, 22, 24]])
      for i in s_1
        p_hat[1] += freq_sop[i]
      end
      #@views p_hat[2] = sum(freq_sop[[2, 5, 7, 9, 16, 18, 20, 23]])
      for i in s_2
        p_hat[2] += freq_sop[i]
      end
      #@views p_hat[3] = sum(freq_sop[[4, 6, 10, 12, 13, 15, 19, 21]])
      for i in s_3
        p_hat[3] += freq_sop[i]
      end
      p_hat ./= m * n # Divide each element of p_hat by m*n

      @. p_ewma = (1 - lam) * p_ewma + lam * p_hat

      stat = chart_stat_sop(p_ewma, chart_choice)

      # Reset win and freq_sop
      fill!(win, 0)
      fill!(freq_sop, 0)
      fill!(p_hat, 0)
    end

    rls[r] = rl
  end
  return rls
end

function test3(m, n, lookup_array_sop, lam, cl, reps_range, chart_choice, dist)

  # Pre-allocate
  n_sops = 24 # factorial(4)
  freq_sop = zeros(Int, n_sops)
  win = zeros(Int, 4)
  data_tmp = zeros(m + 1, n + 1) # empty_data(m, n, dist)
  p_ewma = zeros(3)
  p_hat = zeros(3)
  rls = zeros(Int, length(reps_range))
  sop = zeros(4)

  # Pre-allocate for sum of frequencies
  s_1 = [1, 3, 8, 11, 14, 17, 22, 24]
  s_2 = [2, 5, 7, 9, 16, 18, 20, 23]
  s_3 = [4, 6, 10, 12, 13, 15, 19, 21]

  for r in 1:length(reps_range)
    fill!(p_ewma, 1.0 / 3.0)
    stat = chart_stat_sop(p_ewma, chart_choice)

    rl = 0

    for i in 1:10#while abs(stat) < cl
      rl += 1

      # Fill data 
      rand!(dist, data_tmp)

      # Add noise when using count data
      if dist isa DiscreteUnivariateDistribution
        for j in 1:size(data_tmp, 2)
          for i in 1:size(data_tmp, 1)
            data_tmp[i, j] = data_tmp[i, j] + rand()
          end
        end
      end

      # Compute frequencies of SOPs
      sop_frequencies!(m, n, lookup_array_sop, data_tmp, sop, win, freq_sop)

      @views p_hat[1] = sum(freq_sop[[1, 3, 8, 11, 14, 17, 22, 24]])
      #  for i in s_1
      #    p_hat[1] += freq_sop[i] 
      #  end
      @views p_hat[2] = sum(freq_sop[[2, 5, 7, 9, 16, 18, 20, 23]])
      #  for i in s_2
      #    p_hat[2] += freq_sop[i] 
      #  end
      @views p_hat[3] = sum(freq_sop[[4, 6, 10, 12, 13, 15, 19, 21]])
      #  for i in s_3
      #    p_hat[3] += freq_sop[i] 
      #  end
      p_hat ./= m * n # Divide each element of p_hat by m*n

      @. p_ewma = (1 - lam) * p_ewma + lam * p_hat

      stat = chart_stat_sop(p_ewma, chart_choice)

      # Reset win and freq_sop
      fill!(win, 0)
      fill!(freq_sop, 0)
      fill!(p_hat, 0)
    end

    rls[r] = rl
  end
  return rls
end

#--- Function to create empty matrix which will be filled (either via Monte Carlo or Bootstrap)
empty_data = function (m, n, dist)

  if dist isa Distribution
    data = zeros(m + 1, n + 1)
  else
    data = zeros(size(dist[:, :, 1]))
  end

  return data

end

m = 10
n = 10
lookup_array_sop = compute_lookup_array()
lam = 0.1
cl = 0.03049
dist = Normal(0, 1)
chart_choice = 1
reps_range = 1:1000
@btime test3($m, $n, $lookup_array_sop, $lam, $cl, $reps_range, $chart_choice, $dist)

using Combinatorics

# Define the range of numbers
numbers = 1:9

# Generate all possible 3x3 matrices (flattened as 9-element vectors)
all_combinations = collect(permutations(numbers))
all_combinations = reshape.(all_combinations, 3, 3)

S1 =
  [
    [1 2 3; 4 5 6; 7 8 9],
    [1 4 7; 2 5 8; 3 6 9],
    [3 2 1; 6 5 4; 9 8 7],
    [3 6 9; 2 5 8; 1 4 7],
    [7 4 1; 8 5 2; 9 6 3],
    [7 8 9; 4 5 6; 1 2 3],
    [9 6 3; 8 5 2; 7 4 1],
    [9 8 7; 6 5 4; 3 2 1]
  ]

  S2 = 
  [
    
  ]

all_vec = []


l1_norm = zeros(Int, axes(all_combinations, 1))

for j in axes(S1, 1)
  # Compute the L1 norm of each matrix
  for (i, mat) in enumerate(all_combinations)
    l1_norm[i] = sum(abs.(mat .- S1[j]))
  end


  for i in 1:6
    all_vec = [all_vec; all_combinations[l1_norm.==i]]
  end


end

for i in 1:6
  println(length(all_combinations[l1_norm.==i]))
end


# Function to check monotonic behavior along rows and columns
function is_monotonic(matrix)
  # Check row monotonicity
  for row in 1:3
    if any(matrix[row, i] > matrix[row, i+1] for i in 1:2)
      return false
    end
  end
  # Check column monotonicity
  for col in 1:3
    if any(matrix[i, col] > matrix[i+1, col] for i in 1:2)
      return false
    end
  end
  return true
end

# Filter matrices with monotonic behavior
monotonic_matrices = []
for combination in all_combinations
  mat = reshape(collect(combination), (3, 3))
  if is_monotonic(mat)
    push!(monotonic_matrices, mat)
  end
end

# Display the result
println("Number of monotonic 3x3 matrices: ", length(monotonic_matrices))
for mat in monotonic_matrices
  println(mat)
end