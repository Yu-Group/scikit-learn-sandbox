library(randomForest)

RF_benchmarks <- function(features, responses,
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
library(parallel)
library(doMC)

registerDoMC()
n_cores = 7
options(cores=n_cores)
n.tree.per.core = 30


iRF_benchmarks <- function(features, responses, n_trials=10,
                   K=5,
                   train_split_propn=0.8,
                   n_estimators=20,
                   B=30,
                   M=20,
                   max_depth=5,
                   noisy_split=False,
                   num_splits=2,
                   seed=2018){
      # set seed
      set.seed(seed)

       N_obs <- dim(features)[1]

       # split into testing and training
       training_index <- sample(N_obs, round(train_split_propn * N_obs), replace = FALSE)

       X_train <- as.matrix(features[training_index, ])
       X_test <- as.matrix(features[-training_index, ])

       y_train <- as.factor(responses[training_index])
       y_test <- as.factor(responses[-training_index])

       # again... something about parallelizing
       rfpar <- foreach(n.tree=rep(n.tree.per.core, n_cores), .combine=combine, .multicombine=TRUE)%dopar%{
          randomForest(x=as.matrix(features[training_index, ]), y=as.factor(responses[training_index]), ntree=20)
       }

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
                     , n_core = 7
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
                     feature_importance = feature_importance, stability_all = stability_all, ff = ff)
       return(output)
   }
