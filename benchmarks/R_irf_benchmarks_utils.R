library(randomForest)
library(iRF)
library(parallel)
library(doMC)
n.cores <- 3
registerDoMC(n.cores)

library(plotrix)


get_RF_benchmarks <- function(features, responses,
                          n_trials=10,
                          train_split_propn=0.8,
                          n_estimators=20,
                          seed=2017){
    # set seed
    set.seed(seed)

    N_obs <- dim(features)[1]

    # split into testing and training
    training_index <- sample(N_obs, round(train_split_propn * N_obs), replace = FALSE)

    X_train <- features[training_index, ]
    X_test <- features[-training_index, ]

    y_train <- responses[training_index]
    y_test <- responses[-training_index]

    t_rf <- rep(0,n_trials)
    score <- rep(0,n_trials)
    OOB_error <- rep(0,n_trials)
    class1_error <- rep(0,n_trials)
    class2_error <- rep(0,n_trials)
    feature_importance <- list()

    for(i in 1:n_trials){
        # fit random forest
        t0 <- Sys.time()
        rf <- randomForest(y=y_train, x=X_train, ntree = n_estimators)
        t_rf[i] <- Sys.time() - t0

        # metrics:
        score[i] <- mean(predict(rf, X_test) == y_test)
        OOB_error[i] <- rf$err.rate[dim(rf$err.rate)[1] , 1]
        class1_error[i] <- rf$err.rate[dim(rf$err.rate)[1] , 2]
        class2_error[i] <- rf$err.rate[dim(rf$err.rate)[1] , 3]

        # feature importances:
        feature_importance[[i]] <- importance(rf) # mean decrease in gini

        }


    metrics_all <- list(times = t_rf, score = score, OOB_error = OOB_error, class1_error = class1_error,
                       class2_error = class2_error)

    metrics_summary <- lapply(metrics_all, function(x){c(mean(x), sd(x))})

    output = list(metrics_all = metrics_all, metrics_summary = metrics_summary,
                  feature_importance = feature_importance, rf = rf)
    return(output)
}

# tbh not too sure what the below does... something about parallelizing
#registerDoMC()
#n_cores = 7
#options(cores=n_cores)
#n.tree.per.core = 30

get_iRF_benchmarks <- function(X_train, X_test, y_train, y_test,
                  n_trials=10,
                   K=5,
                   n_estimators=20,
                   B=30,
                   M=20,
                   max_depth=5,
                   noisy_split=False,
                   num_splits=2,
                   seed=2018){
  # set seed
  set.seed(seed)

  # again... something about parallelizing
  #rfpar <- foreach(n.tree=rep(n.tree.per.core, n_cores), .combine=combine, .multicombine=TRUE)%dopar%{
  #randomForest(x=as.matrix(features[training_index, ]), y=as.factor(responses[training_index]), ntree=20)
  #}

  t_iRF <- rep(0,n_trials)
  score <- rep(0,n_trials)
  OOB_error <- rep(0,n_trials)
  class1_error <- rep(0,n_trials)
  class2_error <- rep(0,n_trials)
  feature_importance <- list()
  stability_all <- list()

  for(i in 1:n_trials){
    # fit iRF
    t0 <- Sys.time()
    ff <- iRF(x=X_train
             , y=y_train
             , xtest=X_test
             , ytest=y_test
             , n_iter = K
             , ntree = n_estimators
             , n_core = 3
             , find_interaction=TRUE
             , class_id = 1
             , cutoff_nodesize_prop = 0.1
             , n_bootstrap=B
             , verbose=TRUE
              )
    t_iRF[i] <- Sys.time() - t0

    # metric:
    score[i] <- mean(predict(ff$rf_list[[K]], X_test) == y_test)

    # feature importances:
    feature_importance[[i]] <- importance(ff$rf_list[[K]]) # mean decrease in gini

    # stability scores
    stability_all[[i]] <- ff$interaction[[K]]
    }

   metrics_all <- list(times = t_iRF, score = score)

   metrics_summary <- lapply(metrics_all, function(x){c(mean(x), sd(x))})

   output = list(metrics_all = metrics_all, metrics_summary = metrics_summary,
                 feature_importance = feature_importance, stability_all = stability_all, rf = ff$rf_list[[K]])
   return(output)
}

consolidate_bm_iRF <- function(features, responses, specs,
                seed_data_split = None, seed_classifier = None){

    spec_comb <- expand.grid(specs)
    print(spec_comb)

    iRF_bm = list()

    for(i in 1:dim(spec_comb)[1]){

      print(spec_comb[i,])

      data <- parse_data(features, responses,
        train_split_propn = spec_comb[i,'train_split_propn'],
        N_obs = spec_comb[i,'N_obs'], N_features = spec_comb[i,'N_obs'], seed = 2018)

      iRF_bm[[i]] <- get_iRF_benchmarks(data$X_train, data$X_test,
                        data$y_train, data$y_test,
                        n_trials=spec_comb[i,'n_trials'],
                         K=spec_comb[i,'n_iter'],
                         n_estimators=spec_comb[i,'n_estimators'],
                         B=spec_comb[i, 'n_bootstraps'],
                         M=spec_comb[i,'n_RIT'],
                         max_depth=spec_comb[i,'max_depth'],
                         noisy_split=spec_comb[i,'noisy_split'],
                         num_splits=spec_comb[i,'num_splits'],
                         seed=2018)
      }

    return(iRF_bm)
    }


parse_data <-function(features, responses, train_split_propn = 0.8,
                N_obs = 'all', N_features = 'all', seed = 2018){
  set.seed(seed)

  N = dim(features)[1]
  P = dim(features)[2]

  # subsample data if N_obs parameter is passed
  if(N_obs == 'all'){
      features_subset <- features
      responses_subset <- responses
  }   else{
      indx_subset <- sample(1:N, N_obs)
      features_subset <- features[indx_subset, ]
      responses_subset <- responses[indx_subset]
      }

  # subsample features if p parameter is passed
  if(N_features != 'all'){
      indx_subset <- sample(1:P, N_features)
      features_subset <- features_subset[,indx_subset]
      }

  # split into testing and training\
  train_indx <- sample(1:dim(features_subset)[1],
                  floor(dim(features_subset)[1] * train_split_propn))
  X_train <- features_subset[train_indx, ]
  y_train <- responses_subset[train_indx]
  X_test <- features_subset[-train_indx, ]
  y_test <- responses_subset[-train_indx]

  return(list(
    X_train = X_train, X_test = X_test, y_train = y_train, y_test = y_test))
  }

library(repr)

plot_bm <- function(irf_bm, specs, param, metric){
      x <- specs[[param]]
      y <- rep(0,length(x))
      sd <- rep(0,length(x))

      for(i in 1:length(y)){
          y[i] <- irf_bm[[i]][['metrics_summary']][[metric]][1]
          sd[i] <- irf_bm[[i]][['metrics_summary']][[metric]][2]
      }

      # Change plot size to 4 x 3
      options(repr.plot.width=4, repr.plot.height=4)
      plotCI(x, y, sd, ylab = metric, xlab = param)
  }
