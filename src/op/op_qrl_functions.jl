#--- Function to compute average run length for ordinal patterns
function qrl_op(lam, cl, op_dgp, reps=10000; chart_choice, q_tile=[0.5], d=1, ced=false, ad=100)
         
    # Compute lookup array and number of ops
    lookup_array_op = compute_lookup_array_op()
  
    # Check whether to use threading or multi processing --> only one process threading, else distributed
    if nprocs() == 1
  
      # Make chunks for separate tasks (based on number of threads)        
      chunks = Iterators.partition(1:reps, div(reps, Threads.nthreads())) |> collect
  
      # Run tasks: "Threads.@spawn" for threading, "pmap()" for multiprocessing
      par_results = map(chunks) do i
  
        Threads.@spawn rl_op(lam, cl, lookup_array_op, i, op_dgp, op_dgp.dist, chart_choice; d=d, ced=ced, ad=ad)
  
      end
  
    elseif nprocs() > 1
  
      # Make chunks for separate tasks (based on number of workers)
      chunks = Iterators.partition(1:reps, div(reps, nworkers())) |> collect
  
      par_results = pmap(chunks) do i
        rl_op(lam, cl, lookup_array_op, i, op_dgp, op_dgp.dist, chart_choice; d=d, ced=ced, ad=ad)
      end
  
    end
  
    # Collect results from tasks
    rls = fetch.(par_results)
    rlvec = Iterators.flatten(rls) |> collect
    return quantile(rlvec, q_tile)
  end
  