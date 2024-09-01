set.seed(1)

Flags <- flags(
  flag_numeric("nodes1", 64),
  flag_numeric("nodes2", 64),
  flag_numeric("nodes3", 32),
  flag_numeric("batch_size", 32),
  flag_string("activation1", "relu"),
  flag_string("activation2", "relu"),
  flag_string("activation3", "relu"),
  flag_numeric("learning_rate", .01),
  flag_numeric("dropout", .1)
)

model1 <- keras_model_sequential() %>%
  layer_dense(units = Flags$nodes1, activation = Flags$activation1) %>% 
  layer_dropout(Flags$dropout) %>%
  layer_dense(units = Flags$nodes2, activation = Flags$activation2) %>%
  layer_dropout(Flags$dropout) %>%
  layer_dense(units = Flags$nodes3, activation = Flags$activation3) %>%
  layer_dropout(Flags$dropout) %>%
  layer_dense(units = 1, activation = "sigmoid")

model1 %>% compile(
  optimizer = optimizer_adam(learning_rate = Flags$learning_rate),
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

callbacks = list(callback_early_stopping(monitor = "val_loss", patience = 10, restore_best_weights = TRUE))

model1 %>% fit(
  as.matrix(subset(train_tfidf, select = -target)),
  train_tfidf$target,
  epochs = 300,
  callbacks = callbacks,
  batch_size = Flags$batch_size,
  verbose = FALSE,
  validation_data = list(as.matrix(subset(val_tfidf, select = -target)), val_tfidf$target)
)