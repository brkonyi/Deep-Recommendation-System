#Setup:
    1. The first thing you'll need to do (assuming you haven't already) is install Torch, following the instructions found here: http://torch.ch/docs/getting-started.html.
    2. Grab the code from the repository: https://github.com/brkonyi/Deep-Recommendation-System.
    3. I created a custom criterion/loss function that was used during the original experiments (found in the AbsMultiMarginCriterion folder), so that will need to be installed and built using the Torch install script. To install, move the following files to the listed locations:
        * AbsMultiMarginCriterion.lua, init.c, init.lua -> $TORCH_INSTALL/extra/nn/
        * AbsMultiMarginCriterion.c -> $TORCH_INSTALL/extra/nn/generic/
        * AbsMultiMarginCriterion.cu, init.cu, utils.h -> $TORCH_INSTALL/extra/cunn/
    4. Run install.sh to rebuild Torch with the new criterions. At this point, you should be able to run the script. The script used for the initial experiments is called recommender_network.lua.
    5. Navigate to the dataset/training_set/ directory in the repository and create two new directories named "training" and "validation". This is where the training and validation sets will be stored.
    6. Run the split_dataset.py script. Double check the script to make sure that the maxFiles parameter is set to 500. This will create a training and validation set based on the first 500 movies in the dataset, which is what was used during the initial experiments.
    7. Back in the root of the respository, you should be able to run the following without any errors: th recommender_network.lua --train. This will train whatever network is currently configured in the script for a single epoch and exit, so this is a relatively quick way to check that everything is working.

#Reproduction:
I've attached a spreadsheet of all of the experiments I worked on, but only the last 8 (experiments 7-14) are results that were used. This information can be used to reconfigure the network architecture (found around line 477), as well as the sizes of each layer, as well as the sizes of user and movie vectors (defined at the top of the script). If you have any questions about configuring the network, just let me know and I can help you out.


#Previous Configurations
Here are some things we tried but ended up changing:
* Originally, we treated this as a regression problem. However, since ratings in the dataset are integers in the range of 1-5, it didn't make much sense to allow the network to predict real values between 1-5, so we switched to using a classification scheme for the network instead.
* Originally we used a few different loss functions (MSE, L1), but we ran into some issues since the dataset was unbalanced. We then switched to a variation of MultiMarginCriterion (https://github.com/torch/nn/blob/master/doc/criterion.md#nn.MultiMarginCriterion) to deal with the unbalanced dataset and to assign greater losses to predicted ratings that are further from the true rating. This is described in more detail in the attached document and slide deck.

#Future Improvements:
I'm currently working on the following:
* To deal with the low number of training examples per user, we're creating an alternative representation of user vector encodings based on the movie vector encodings of the movies the user has watched. This code is all found in recommender_network_alt_user_rep.lua.
* Creating a new loss function based on the negative log likelihood loss function (https://github.com/torch/nn/blob/master/doc/criterion.md#classnllcriterion) to more heavily penalize poor ratings. I'm currently running some trials with the vanilla ClassNLLCriterion and validation RMSE is already looking better than it was with the AbsMultiMarginCriterion.
* Adding dropout layers with 50% chance of dropout between each layer to deal with some overfitting we're seeing.
* Training the updated network configuration on the full Netflix dataset. I couldn't do this before since LuaJIT has some weird memory limits which require you to write special structures for large amounts of data.
