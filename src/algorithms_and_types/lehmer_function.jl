export perm_to_lehm_idx!,
  perm_to_lehm_idx

"""
    perm_to_lehm_idx(P::Vector{<:Integer}) -> Int

Converts a permutation vector `P` into its 0-based lexicographical index. This is the in-place
version.

# Arguments
- `P`: A permutation vector (e.g., [3, 1, 2]).
- `used`: A vector to track used elements (should be initialized to zeros of length equal to the maximum element in `P`).

# Returns
- The 0-based index of the permutation.
"""
function perm_to_lehm_idx(
  P::Vector{<:Integer}
)::Int

  # Vector to keep track of used elements
  used = zeros(Int, length(P))
  # Length of the permutation
  n = length(P)
  # Initialize index
  index = 0

  # Iterate through the permutation
  for i in 1:n
    current_element = P[i]
    coefficient = 0

    # Count how many smaller numbers have not been used
    for j in 1:(current_element-1)
      if used[j] == 0
        coefficient += 1
      end
    end

    # Add weighted coefficient to index (factorial number system)
    rank = n - i
    index += coefficient * factorial(rank)

    # Mark current element as used
    used[current_element] = 1
  end

  return index + 1  # 1-based indexing
end


"""
    perm_to_lehm_idx!(P::Vector{<:Integer}) -> Int

Converts a permutation vector `P` into its 0-based lexicographical index. This is the in-place
version.

# Arguments
- `P`: A permutation vector (e.g., [3, 1, 2]).
- `used`: A vector to track used elements (should be initialized to zeros of length equal to the maximum element in `P`).

# Returns
- The 0-based index of the permutation.
"""
function perm_to_lehm_idx!(
  P::Vector{<:Integer},
  used::Vector{<:Integer}
)::Int

  # Length of the permutation
  n = length(P)
  index = 0

  # Iterate through the permutation
  for i in 1:n
    current_element = P[i]
    coefficient = 0

    # Count how many smaller numbers have not been used
    for j in 1:(current_element-1)
      if used[j] == 0
        coefficient += 1
      end
    end

    # Add weighted coefficient to index (factorial number system)
    rank = n - i
    index += coefficient * factorial(rank)

    # Mark current element as used
    used[current_element] = 1
  end

  return index + 1  # 1-based indexing
end



