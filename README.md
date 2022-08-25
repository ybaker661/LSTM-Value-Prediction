# LSTM-Value-Prediction
Notebooks, Code, and Models Associated with LSTM Based Value Function Prediction

This repository contains the code for the above mentioned task, which include two separate jupyter notebooks for the real time and hour ahead prediction tasks
respectively, as well as their associated utils files. The real time and hour ahead utils files are largely the same, save for the hour shift in the functions
of the H.A. util files.

The jupyter notebooks are labeled and fully parametrized for ease of running trials. The code is organized generally as follows:
data loading and prep -> parameter setting -> valuation -> training -> evaluation (prediction, and arbitrage)

similarly, the utils files are split into three main "sections"

valuation and arbitrage functions (based on arbval and valuation functions on matlab)
dataset functions (for preproccesing price data into dataset for training)
network function (for creating the NN model)
evaluation functions (based almost 1:1 from the arbsim functions on matlab, except there is preprocessing step based on the generate_train function to allow the
network to predict using the testing price data)

The Hour Ahead notebook is longer since it also includes hourly valuation, as well as the tests for Houston and Queensland.

Please note that in the training subheaders in the notebook, you'll find the variable test formatted as:

test = 'CNN_LSTM_' + str(num_segment) +'_12hr_HA' 

here the 12hr can be subbed out for 0.25 and 0.5 corresponding to the 4 hour and 2 hour energy storage duration cases. Further, there is a variable called net
which should be set to "t1" or "vanilla" based on which network you would like to instantiate (the networks are exactly the same, save for one MaxPool layer)

The full results including raw profit values and profit ratios can be found in the two excel files (for hour ahead, HA, and real time, RT, respectively)

Please run the code cells sequentially, however when running different tests for the same region, all you need to do is rerun the parameter cell with your new
settings, and then run the remaining code cells of interest once you've made sure the parameters have been correctly updated.
