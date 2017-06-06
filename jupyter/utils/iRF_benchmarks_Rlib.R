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
    train_split_propn=0.9
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
