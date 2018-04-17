dataSource = read.csv("train_source.txt", row.names = FALSE)
dataTarget = read.csv("train_target.txt", row.names = FALSE)
percentages = c(0.01, 0.05, 0.10, 0.20, 0.40, 0.60, 0,80)

for(i in percentages) {
  cutoff = round(i * nrow(dataSource))
  blockSource = dataSource[1:cutoff,]
  blockTarget = dataTarget[1:cutoff,]
  write.table(blockSource, sprintf("train_source%f.txt", i), row.names = FALSE)
  write.table(blockTarget, sprintf("train_target.txt", i), row.names = FALSE)
}