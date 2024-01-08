players <- read.csv("players.csv")
plays <- read.csv("plays.csv")
tackles <- read.csv("tackles.csv")
tracking_week_1 <- read.csv("tracking_week_1.csv")
tracking_week_2 <- read.csv("tracking_week_2.csv")
tracking_week_3 <- read.csv("tracking_week_3.csv")
tracking_week_4 <- read.csv("tracking_week_4.csv")
tracking_week_5 <- read.csv("tracking_week_5.csv")
tracking_week_6 <- read.csv("tracking_week_6.csv")
tracking_week_7 <- read.csv("tracking_week_7.csv")
tracking_week_8 <- read.csv("tracking_week_8.csv")
tracking_week_9 <- read.csv("tracking_week_9.csv")
library(dplyr)
library(cluster)
library(ggplot2 )
library(clValid)
library(tidymodels)
library(SuperLearner)
library(caret)
library(gt)

all_tracking <- rbind(tracking_week_1, tracking_week_2, tracking_week_3, tracking_week_4,
                      tracking_week_5, tracking_week_6, tracking_week_7, tracking_week_8,
                      tracking_week_9)

plays_tackles <-tackles %>% 
  left_join(plays, by = c("gameId","playId")) %>% 
  rename(tackling_playerId = nflId)

plays_pure_tackles <-tackles %>% 
  filter(tackle == 1) %>% 
  left_join(plays, by = c("gameId","playId")) %>% 
  rename(tackling_playerId = nflId)
  
#tackles epa
tackled <- plays_tackles %>% 
  filter(tackle == 1) %>% 
  group_by(tackling_playerId)  %>% 
  summarize(epa_tackle = mean(expectedPointsAdded), tackles = n()) %>% 
  left_join(players, by = c("tackling_playerId" = "nflId"))

#assist tackle epa
assist <- plays_tackles %>% 
  filter(assist == 1) %>% 
  group_by(tackling_playerId)  %>% 
  summarize(epa_assist_tackle = mean(expectedPointsAdded), assist_tackles = n())

#missed tackles epa
missed <- plays_tackles %>% 
  filter(pff_missedTackle == 1) %>% 
  group_by(tackling_playerId)  %>% 
  summarize(epa_missed_tackle = mean(expectedPointsAdded), missed_tackles = n())
  


#look at # of plays and tackle rate
player_plays <- all_tracking %>% 
  group_by(gameId,playId,nflId) %>% 
  summarize(frames = n()) %>% 
  group_by(nflId) %>% 
  summarize(plays = n())

tackling_stats <- player_plays %>% 
  left_join(tackled, by = c("nflId" = "tackling_playerId")) %>% 
  inner_join(missed, by = c("nflId" = "tackling_playerId")) %>% 
  left_join(assist, by = c("nflId" = "tackling_playerId")) %>% 
  mutate_all(~ifelse(is.na(.), 0, .)) %>% #it might not be best to use 0, consider imputing values
  mutate(weighted_tackles = tackles - missed_tackles + 0.5*assist_tackles,
         pure_tackle_rate = tackles/plays, missed_tackle_rate = missed_tackles/plays,
         assist_tackle_rate = assist_tackles/plays, weighted_tackle_rate = weighted_tackles/plays,
        miss_to_tackle_ratio = tackles/missed_tackles)

tackling_stats %>% 
  group_by(position) %>% 
  summarize(epa_tackle_pos =  mean(epa_tackle)) %>% 
  filter(epa_tackle_pos != 0) %>% 
  ggplot(aes(x = reorder(position,-epa_tackle_pos), y = epa_tackle_pos))+
  geom_bar(stat = "identity")+
  labs(x = "Position", y = "EPA Per Tackle", title = "EPA Per Tackle by Position")
ggsave("EPAbyPos.png", width = 10, height =8, dpi = "retina")


point_of <- all_tracking %>% 
  filter(event %in% c("pass_outcome_caught","run","handoff")) %>% 
  rename(trackingId = nflId)

#cut down fields
point_of_plays <- point_of %>% 
  left_join(plays_pure_tackles,by = c("gameId","playId")) %>% 
  filter(playNullifiedByPenalty != "Y") %>% 
  select(-expectedPoints,-expectedPointsAdded,-frameId,-homeTeamWinProbabilityAdded,
         -visitorTeamWinProbilityAdded,-preSnapHomeTeamWinProbability,-preSnapVisitorTeamWinProbability,
         -playNullifiedByPenalty,-yardlineNumber,-yardlineSide,-penaltyYards,
         -foulName1,-foulName2,-foulNFLId1,-foulNFLId2)

#separate so it doesn't take as long to run
point_of_plays <- point_of_plays %>% 
  filter(!is.na(trackingId)) %>% 
  mutate(tackler = ifelse(trackingId == tackling_playerId,1,0), 
         carrier = ifelse(trackingId == ballCarrierId,1,0), 
         unique_id = paste(gameId,playId, sep = ""))

point_of_plays<- point_of_plays %>% 
  filter(carrier == 1) %>%
  select(gameId,playId,x,y,s,a,o,dir) %>% 
  rename_with(~ paste0("carrier_", .), everything()) %>% 
  right_join(point_of_plays, by = c("carrier_gameId" = "gameId", "carrier_playId" = "playId"))

#create model data set
tackling_model_data <- point_of_plays %>% 
  mutate(distCarrier = sqrt((x-carrier_x)^2+(y-carrier_y)^2),
         rel_speed = s-carrier_s,
         rel_a = a-carrier_a,
         rel_o = o - carrier_o,
         rel_dir = dir-carrier_dir,
         rel_x = x-carrier_x,
         rel_y = x-carrier_y,
         tackler = as.factor(tackler),
         playDirection = as.factor(playDirection),
         quarter = as.factor(quarter),
         event = as.factor(event),
         down = as.factor(down)) %>% 
  filter(club == defensiveTeam) %>% 
  select(playDirection,tackler,distCarrier,rel_speed,rel_a,defendersInTheBox,
         event, offenseFormation, passProbability) #should we include things not necessarily related to player location like formation

tackling_model_data %>% 
  filter(tackler == 1) %>% 
  with(hist(distCarrier, xlab = "Distance to Carrier",ylab = "Frequency", 
            main = "Distance to Carrier Histogram"))


#training/testing data  
split <- initial_split(tackling_model_data,prop = 0.8, strata = tackler) #strata ensures proportion of tacklers is even in both the train and test data
train <- split %>% 
  training()

test <- split %>% 
  testing()


#tune parameters of model
log_reg <- logistic_reg(mixture = tune(), penalty = tune(), engine = "glmnet") 
grid <- grid_regular(mixture(), penalty(), levels = c(mixture = 4, penalty = 3))   #levels sets values?
#sets up workflow
log_reg_wf <- workflow() %>% 
  add_model(log_reg) %>% 
  add_formula(tackler~.)

#creates cross validation
folds <- vfold_cv(train, v= 5)

#runs models
log_reg_tuned <- tune_grid(
  log_reg_ef,
  resamples =  folds,
  grid = grid,
  control = control_grid(save_pred = TRUE))

#selects best model based on roc_auc
features <- select_best(log_reg_tuned, metric = "roc_auc")

#final model run based on select_best
log_reg_final <- logistic_reg(mixture = features$mixture[1], penalty = features$penalty[1]) %>%
  set_engine("glmnet") %>% 
  set_mode("classification") %>% 
  fit(tackler~., data =train)

#predictions
pred_class <- predict(log_reg_final, new_data = test, type = "class")
pred_prob <- predict(log_reg_final, new_data = test, type = "prob")
pred_prob[,2]
results <- test %>% 
  select(tackler) %>% 
  bind_cols(pred_class,pred_prob )
confusionMatrix(results$.pred_class,test$tackler) #confusion matrix
loprecision(results, truth = tackler,
          estimate = .pred_class)
recall(results, truth = tackler,
       estimate = .pred_class)

#variable importance, do more research into feature selection
coeff <- tidy(log_reg_final) %>% 
  arrange(desc(abs(estimate)))

tackles_over <- point_of_plays %>% 
  mutate(distCarrier = sqrt((x-carrier_x)^2+(y-carrier_y)^2),
         rel_speed = s-carrier_s,
         rel_a = a-carrier_a,
         rel_o = o - carrier_o,
         rel_dir = dir-carrier_dir,
         rel_x = x-carrier_x,
         rel_y = x-carrier_y,
         tackler = as.factor(tackler),
         playDirection = as.factor(playDirection),
         quarter = as.factor(quarter),
         event = as.factor(event),
         down = as.factor(down)) %>% 
  filter(club == defensiveTeam) 

tackles_over$tackle_prob <-predict(log_reg_final,new_data = tackles_over %>% 
  select(playDirection,tackler,distCarrier,rel_speed,rel_a,defendersInTheBox,
         event, offenseFormation, passProbability), type = "prob")$.pred_1

tackles_over <- tackles_over %>% 
  group_by(unique_id) %>% 
  mutate(scaled_tackle_prob = tackle_prob/sum(tackle_prob)) %>% 
  ungroup() %>% 
  mutate(toe = (as.numeric(tackler)-1)-scaled_tackle_prob)

toe_results <- tackles_over %>% 
  group_by(trackingId) %>% 
  summarize(displayName = first(displayName), toe_play = mean(toe), total_toe = sum(toe) ,plays = n(), total_tackle_prob = sum(scaled_tackle_prob),
            tackles = sum(as.numeric(tackler)-1), tackle_rate = mean(as.numeric(tackler)-1))
toe_results %>% 
  filter(plays>=50) %>% 
  top_n(10, toe_play) %>%
  ggplot(aes(x = reorder(displayName, -toe_play), y = toe_play))+
  geom_bar(stat = "identity")+
  labs(title = "Top 10 Players by TOE Per Play", subtitle = "Minimum 50 Plays",
       x = "", y = "Tackle Over Expectation Per Play")+
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggsave("TopTOE.png", width = 10, height =8, dpi = "retina")

solo_tackle_stats <-toe_results %>% 
  inner_join(tackled, by = c("trackingId"="tackling_playerId")) #figure out why tackle stats don't matc

#Clustering
clustering_data <- solo_tackle_stats %>% 
  filter(plays>=50) %>% 
  select(toe,epa_tackle,tackle_rate) %>% 
  na.omit()

clustering_data <- scale(clustering_data)

# Decide how many clusters to look at
n_clusters <- 10

# Initialize total within sum of squares error: wss
wss <- numeric(n_clusters)

set.seed(123)

# Look over 1 to n possible clusters
for (i in 1:n_clusters) {
  # Fit the model: km.out
  km.out <- kmeans(clustering_data, centers = i, nstart = 20)
  # Save the within cluster sum of squares
  wss[i] <- km.out$tot.withinss
}

# Produce a scree plot
wss_df <- tibble(clusters = 1:n_clusters, wss = wss)

scree_plot <- ggplot(wss_df, aes(x = clusters, y = wss, group = 1)) +
  geom_point(size = 4)+
  geom_line() +
  scale_x_continuous(breaks = c(2, 4, 6, 8, 10)) +
  xlab('Number of clusters')
scree_plot+
  geom_hline(
    yintercept = wss, 
    linetype = 'dashed'
  )

model=kmeans(clustering_data,centers = 4, nstart =20)

cluster_breakdown <-solo_tackle_stats %>% 
  filter(!is.na(epa_tackle),!is.na(tackle_rate), !is.na(toe))%>% 
  filter(plays >= 50) %>% 
  mutate(cluster = ifelse(model$cluster == 1,"High Volume/Low Impact",
                          ifelse(model$cluster == 2, "Low Volume/High Impact",
                                 ifelse(model$cluster == 3, "Average Volume/Above Average Impact",
                                        "Average Volume/Low Impact")))) %>%
  group_by(cluster,position) %>% 
  summarize(count = n()) %>% 
  pivot_wider(names_from = position, values_from = count) %>% 
  mutate_all(~replace(., is.na(.), 0)) %>% 
  ungroup() %>% 
  gt() %>% 
  tab_header(
    title = md("Cluster Breakdown by Position"),
    subtitle = md("Minimum 50 Plays")
  ) %>% 
  gtExtras::gt_theme_538()
gtsave(cluster_breakdown, "ClusterBreakdown.png") 


solo_tackle_stats %>% 
  filter(!is.na(epa_tackle),!is.na(tackle_rate), !is.na(toe))%>% 
  filter(plays >= 50) %>% 
  mutate(cluster = ifelse(model$cluster == 1,"High Volume/Low Impact",
                          ifelse(model$cluster == 2, "Low Volume/High Impact",
                                 ifelse(model$cluster == 3, "Average Volume/Above Average Impact",
                                        "Average Volume/Low Impact")))) %>% 
  select(displayName.x,toe,tackle_rate, epa_tackle, cluster) %>% 
  mutate_if(is.numeric, ~round(., 2)) %>% 
  gt() %>% 
  cols_label(displayName.x = "Player Name",
             toe = "TOE Per Play",
             tackle_rate = "Tackle Rate",
             epa_tackle = "EPA Per Tackle") %>% 
  tab_header(
    title = md("Tackling Overview")
  ) %>% 
  gtExtras::gt_theme_538()
