import PrecompileTools

PrecompileTools.@compile_workload begin

  # OP functions 
  data = rand(100)
  lam = 0.1

  stat_op(data, lam; chart_choice=1, op_length=3, d=1)
  test_op(data; chart_choice=1, op_length=3, d=1, alpha=0.05)

  # SOP functions
  data_sop = rand(10, 10)
  stat_sop(data_sop, 1, 1;
    chart_choice=1, add_noise=false
  )
  stat_sop_bp(
    data_sop, 3; chart_choice=1, add_noise=false
  )
  test_sop(
    data_sop, 0.05, 1, 1; chart_choice=1, refinement=0, add_noise=false, approximate=false
  )

  data_sop = rand(10, 10, 10)
  stat_sop(
    data_sop, 0.1, 1, 1;
    chart_choice=1, add_noise=false
  )
  stat_sop_bp(
    data_sop, 0.1, 3; chart_choice=1, add_noise=false
  )

end