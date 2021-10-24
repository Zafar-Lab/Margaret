library(dyntoy)


dataset = generate_dataset(
  model = model_multifurcating(),
  num_cells = 5000,
  num_features = 1000
)

saveRDS(dataset, file='/home/lexent/Desktop/dyntoy_disconnected_gen_1.rds')
