######################
#Project: Phenome Force Presentation
#Purpose: Code only
#Author: Henri Chung
#Date: Oct 6, 2020
######################

list.of.packages <- c("tidyverse", "tidymodels", "ranger", "kknn", "randomForest")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages, repos = "https://cloud.r-project.org")
invisible(capture.output(lapply(list.of.packages, library, character.only = TRUE, warn.conflicts = FALSE, quietly = T)))
rm(list = ls()) 

#set random seed for reproducability
set.seed(123)
#Read in data
plant_data <- readr::read_csv("pdata.csv") %>% #read in CSV file
  dplyr::select(-c("Trait", "myIndex")) %>% #select traits of interest
  dplyr::mutate(fertilize = ifelse(Plot%%2 ==0, TRUE, FALSE)) %>% #add boolean fertilizer variable
  dplyr::mutate(crop = factor(sample(c("maize", "soybean", "alfalfa"), size = nrow(.), replace = T))) %>% #assign plots different crops
  janitor::clean_names() #reduce column names to lowercase, remove punctuation
colnames(plant_data) #look at column names
head(plant_data) #peek at data

#split the dataset into a training and testing set
plant_split <- rsample::initial_split(plant_data, prop = 0.9) #create a split object where 9/10 proportion of data is used for training
plant_train <- rsample::testing(plant_split) #isolate training set into one object
plant_test <- rsample::training(plant_split) #isolate testing set into one object
tibble::glimpse(plant_test) #peek at plant testing set

plant_recipe <- head(plant_train) %>%
  recipes::recipe(eph ~ ngrdi) 

#if the model formula is extremely long, it may be necessary to define the model formula using update_role functions
plant_recipe2 <- head(plant_train) %>%
  recipes::recipe() %>%
  recipes::update_role(eph, new_role = "outcome") %>% #
  recipes::update_role(ngrdi, new_role = "predictor") 

#"selector functions can be used to select a group of variables at one time
plant_recipe3 <- head(plant_train) %>%
  recipes::recipe() %>%
  recipes::update_role(matches("ep."), new_role = "outcome") %>%
  recipes::update_role(contains("N"), new_role = "predictor")

#you can also yse steps to transform variables with step functions (https://recipes.tidymodels.org/reference/index.html#section-step-functions-individual-transformations)
plant_recipe4 <- head(plant_train) %>%
  recipes::recipe(eph ~ ngrdi) %>%
  recipes::step_center(all_predictors(), -all_outcomes()) %>%
  recipes::step_scale(all_predictors(), -all_outcomes()) 


#execute the recipe on test data to validate
plant_recipe <- recipes::prep(plant_recipe)
#extract data where the recipe was applied 
plant_head <- recipes::juice(plant_recipe)

#apply recipe to data of interest
plant_train <- recipes::bake(plant_recipe, plant_train)
plant_test <- recipes::bake(plant_recipe, plant_test)
tibble::glimpse(plant_test)

#Define model
linear_lm <- linear_reg() %>% #define a linear regression model
  set_engine("lm") %>% #decide what package you want to use for the linear regression
  set_mode("regression")  #set regression or classification depending on model specifications

#alternative model arrangement for comparison
linear_glmnet <- linear_reg() %>%
  set_engine("stan") %>%
  set_mode("regression")

#note engines do not have to be R packages, tidymodels supports interfacing with outside programs such as keras or tensorflow. One of the benefits to using R.

#create a workflow
plant_workflow <- workflows::workflow() %>% #define workflow
  workflows::add_recipe(plant_recipe) %>% #add recipe specification
  workflows::add_model(linear_lm) #add model specification

#fit workflow to data
plant_fit <- parsnip::fit(plant_workflow, plant_train)

#predict test data from fitting model
plant_pred <- predict(plant_fit, plant_test)
#bind prediction results to test data for easy comparison
plant_test <- dplyr::bind_cols(plant_test, plant_pred)
tibble::glimpse(plant_test)

#calculate model performance metrics (default)
plant_metrics <- yardstick::metrics(plant_test, truth = eph, estimate = .pred)
plant_metrics

#calculate a single model performance metric
yardstick::rmse(data = plant_test, truth = eph, estimate = .pred)

#calculate a custom set of model performance metrics
plant_multi <- yardstick::metric_set(rmse, rsq)
plant_multi(data = plant_test, truth = eph, estimate = .pred)

#Split training data for cross validation scheme into 5 folds with 5 repeats
plant_folds <- rsample::vfold_cv(training(plant_split), v = 5, repeats = 5)
#define a control object to change settings in the cross validation process
plant_control <- tune::control_grid(
  verbose = FALSE,  #print out progress while fitting
  allow_par = TRUE, #allow parallel processing
  extract = function(x){extract_model(x)},  #extract the individual fitting model object for each split
  save_pred = TRUE #save model predictions
)

#fit the workflow to the folds object, with control settings
plant_cv <- tune::fit_resamples(plant_workflow, plant_folds, control = plant_control)
#collect performance metrics from the folds
plant_metrics <- tune::collect_metrics(plant_cv)
#collect predictions from the folds
plant_predictions <- tune::collect_predictions(plant_cv)

#Definee a new recipe for random forest model 
rf_recipe <- head(training(plant_split)) %>%
  recipes::recipe(crop ~ .) %>%
  recipes::update_role(name, new_role = "id") %>%
  recipes::step_string2factor(crop) 

#Define a new model 
rf_tune <- parsnip::rand_forest(trees = 100, mtry = tune(), min_n = tune()) %>% #mtry and min_n parameters are set to "tune()"
  parsnip::set_engine("randomForest") %>% #use ranger package
  parsnip::set_mode("classification") #set to classification

#define new workflow with recipe and model
tune_wf <- workflows::workflow() %>%
  workflows::add_recipe(rf_recipe) %>%
  workflows::add_model(rf_tune)

#tune the models
control = control_grid(extract = function (x) extract_model(x)) #extract models used in tuning (WARNING: models may take up a large amount of memory)
tune_res <- tune::tune_grid(tune_wf, resamples = plant_folds, grid = 5, control = control)
head(tune_res)
#Check variable important by mean decrease in Gini impurity
model <- tune_res$.extracts[[1]]$.extracts[[1]] 
model
randomForest::importance(model)


#collect metrics for tuning
tune_metrics <- tune::collect_metrics(tune_res)

#reshape data from wide to long format for plotting
p1data <- tune_metrics %>%
  tidyr::pivot_longer(min_n:mtry, values_to = "value",names_to = "parameter")

#plot data
p1 <- p1data %>%  
  ggplot(aes(x = value, y = mean)) + 
  geom_point() + 
  facet_wrap(parameter~.metric, scales = "free")
p1

#select the best parameters from tuning routine
best_params <- tune::select_best(tune_res, "roc_auc")
#final model using parameters selected from tune results
final_rf <- tune::finalize_model(rf_tune, best_params)

#add final model into workflow
final_wf <- workflow() %>%
  add_recipe(rf_recipe) %>%
  add_model(final_rf)

#fit the finalized model to the testing data 
final_res <- final_wf %>% tune::last_fit(plant_split)
collect_metrics(final_res)

############################################

#set seed to random integer
set.seed(Sys.time())
#create list to store folds
custom_folds <- list()
#create 25 splits of the training data 
for(i in 1:25){
  custom_folds[[i]] <- initial_split(training(plant_split), prop = 0.8)
}
#convert list folds to tibble
custom_folds <- tibble(custom_folds)

#compare object size
pryr::object_size(plant_folds)
pryr::object_size(custom_folds)

###########################################

#Create Sample data
big_mat <- as.data.frame(matrix(rexp(20000, rate=2), ncol=200))
big_mat$var <- as.factor(sample(c(TRUE, FALSE), nrow(big_mat), replace = TRUE))

#Define PCA Recipe
recipe <- head(big_mat) %>%
  recipes::recipe(var ~ .) %>%
  recipes::step_center(all_predictors()) %>%
  recipes::step_scale(all_predictors()) %>%
  step_pca(all_predictors(), num_comp = 2)

#Define KNN model
model <- nearest_neighbor(neighbors = 5, weight_func = "rectangular") %>%
  set_engine("kknn") %>%
  set_mode("classification") 

#Create workflow
workflow <- workflow() %>%
  add_recipe(recipe) %>%
  add_model(model)

#fit model
fit <- fit(workflow, big_mat)
#model predictions
preds <- predict(fit, big_mat)
#bind model predictions to original data
res <- cbind(big_mat, preds) %>%
  select(var, .pred_class) 

#calculate performance metrics
res_metrics <- metrics(res, truth = var, estimate = .pred_class)
