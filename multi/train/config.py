import os

filename = 'stations_xibei.train'
eval_metric = 'PianChaKaoHe'  #choices are: [MSE, MAE, PianChaKaoHe, KouDian_RMSE, KouDian_MAE]



#May not modify below:
loop_days = 1     #do loop of [train and test and save output] day by day (overwrite by latest)
shift_months = 1   #now is now if shift_months=0, else shift backward, e.g. when shift=1 'now' is a month earlier. Usually, we take 1 month shift assuring that both train and test will have gt labels (real power).

train_length = 45   #train days   
test_length = 7  #test days


