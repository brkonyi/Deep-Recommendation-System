------------------|
--PACKAGE IMPORTS |
------------------|
require 'nn'
require 'optim'
require 'cutorch'
require 'cunn'
require 'utils'

local ffi = require 'ffi'
local tds = require 'tds'

--------------------------------|
-- C DATA STRUCTURE DEFINITIONS |
--------------------------------|
ffi.cdef[[
typedef struct { uint32_t userId; uint16_t movieId; uint8_t rating; } review;
typedef struct { uint16_t count; uint16_t* movieIds; uint8_t* ratings; } userMovies;
void free(void*);
void* malloc(size_t);
]]

local review_t = ffi.typeof("review")
local review_p = ffi.typeof("review*")
local userMovies_t = ffi.typeof("userMovies")
local userMovies_p = ffi.typeof("userMovies*")
local uint16_t = ffi.typeof("uint16_t")
local uint16_p = ffi.typeof("uint16_t*")
local uint8_t  = ffi.typeof("uint8_t")
local uint8_p  = ffi.typeof("uint8_t*")

------------------------------|
-- START CONSTANTS DEFINITION |
------------------------------|
DEFAULT_TRAINING_SET_DIR = './dataset/training_set/training/'
DEFAULT_VALIDATION_SET_DIR = './dataset/training_set/validation/'

USER_VEC_LEN  = 32 
MOVIE_VEC_LEN = USER_VEC_LEN
INPUT_VEC_LEN = USER_VEC_LEN + MOVIE_VEC_LEN
FIRST_LAYER_SIZE  = 128 
SECOND_LAYER_SIZE = 192
THIRD_LAYER_SIZE  = 256

LOGGER_TRAINING = "training_data"
LOGGER_VALIDATION = "validation_data"

------------------------------|
-- END CONSTANT DEFINITIONS   |
------------------------------|


--Used to dump accuracy information after training is complete 
function printCSV(config)
    print(string.format("Epoch, Training Accuracy, Validation Accuracy, Training RMSE, Validation RMSE"))
    for i = 1, #config.errors do
        epochError = config.errors[i]
        if epochError.validationError == nil then
            epochError.validationError = 0
            epochError.validationRMSE = 0
        end
        print(string.format("%d, %.4f%%, %.4f%%, %.4f, %.4f", i, 100 - epochError.trainingError, 100 - epochError.validationError, epochError.trainingRMSE, epochError.validationRMSE))
    end 
end

--Parse the raw review and put its information into a table
function createReview(rawReview)
    local splitReview = split(rawReview, ',')
    if tonumber(splitReview[1]) == nil then
        return nil
    end
    local review = ffi.cast(review_p, ffi.C.malloc(ffi.sizeof(review_t)))
    
    review.userId = tonumber(splitReview[1])
    review.rating = tonumber(splitReview[2])
    --Note: movie ID is populated in loadDataset, so no, it wasn't forgotten!
    return review
end

--Returns a movie vector. If there isn't currently a vector for this movie, initialize one.
function getMovie(movieId, config)
    --Check to see if there's a movie vector for this movie id
    if config.movieVecs[movieId] == nil then
       config.movieVecs[movieId] = torch.randn(MOVIE_VEC_LEN, 1):cuda() 
    end
    return config.movieVecs[movieId]
end

--Returns a user vector. If there isn't currently a vector for this user, intialize one.
function getUser(userId, config)
    --Check to see if there's a user vector for this user id
    if config.userVecs[userId] == nil then
        config.userVecs[userId] = torch.randn(USER_VEC_LEN, 1):cuda()
    end
    return config.userVecs[userId]
end

--Parse a review string into a usable review table
function getReview(rawReview)
    if rawReview == nil then
        return nil
    end
    return createReview(rawReview)
end

--Load the dataset into an array of C review structures
function loadDataset(directory, userList)
    if directory == nil then return {} end

    local files = getFileNames(directory)
    local dataset = {}
    local weights = {}
    local maxUser = -1

    for i = 1, 5 do
        weights[i] = 0
    end

    --Find the number of reviews in the dataset
    local reviewCount = 0
    for i = 1, #files do
        local f = io.open(directory..files[i], 'r')
        local movieString = f:read()
        while true do
            local review = getReview(f:read())
            ffi.C.free(review)
            if review == nil then 
                break
            end
            
            reviewCount = reviewCount + 1
        end
        f:close()
        xlua.progress(i, #files)
    end

    --TODO should probably free this at some point, but it lives for the length of the program anyways so not a huge deal
    local reviews = ffi.cast(review_p, ffi.C.malloc(ffi.sizeof(review_t) * (reviewCount + 1)))
    reviewCount = 0

    local newUserList = nil
    if userList == nil then
        newUserList = tds.Hash()
    end

    --Actually do the loading here
    for i = 1, #files do
        local f = io.open(directory..files[i], 'r')
        local movieString = f:read()
        local movieId = tonumber(string.sub(movieString, 0, #movieString - 1))

        while true do
            local review = getReview(f:read())
            if review == nil then break end

            --If the userList isn't nil, we're loading a validation set and we only want to load reviews by users
            --that are seen in our training set. If it is nil, we'll build a list of users to be used later.
            if userList == nil or userList[review.userId] ~= nil then
                reviewCount = reviewCount + 1
                reviews[reviewCount].movieId = movieId
                reviews[reviewCount].userId = review.userId
                reviews[reviewCount].rating = review.rating

                --Keep track of how many reviews of each rating occur
                weights[review.rating] = weights[review.rating] + 1

                --We need to keep track of the max user to allocate our array of user to ratings mappings
                if review.userId > maxUser then
                    maxUser = review.userId
                end

                if userList == nil then
                    newUserList[review.userId] = 1
                end
            end

            ffi.C.free(review)
        end
        f:close()
        xlua.progress(i, #files)
    end

    --Calculate the final weights
    for i = 1, 5 do
        weights[i] = weights[i] / reviewCount 
    end

    --If we created a new list of users we've seen, we want to return it
    if userList == nil then
        userList = newUserList
    end
    
    return reviews, reviewCount, weights, userList, maxUser
end

--Creates a mapping of each user to all the movies they've rated + their ratings
function userRatingsMatch(trainingSet, trainingSetSize, numUsers)
    --TODO Should probably free this when we're done to be clean, but we use it the entire life of the program
    local userMoviesC = ffi.C.malloc(ffi.sizeof(userMovies_t) * (numUsers + 1))
    ffi.fill(userMoviesC, ffi.sizeof(userMovies_t) * (numUsers + 1))
    local userMovies = ffi.cast(userMovies_p, userMoviesC)

    for i = 1, trainingSetSize do
        local user = trainingSet[i].userId
        userMovies[user].count = userMovies[user].count + 1
    end
    
    for i = 1, trainingSetSize do
        local user = trainingSet[i].userId
        local count = userMovies[user].count
        
        if userMovies[user].movieIds == nil then
            --TODO Should also free these eventually, but not a huge deal if we don't
            userMovies[user].movieIds = ffi.cast(uint16_p, ffi.C.malloc(ffi.sizeof(uint16_t) * (count + 1)))
            userMovies[user].ratings = ffi.cast(uint8_p, ffi.C.malloc(ffi.sizeof(uint8_t) * (count + 1)))
            count = 0
        end

        userMovies[user].movieIds[count + 1] = trainingSet[i].movieId
        userMovies[user].ratings[count + 1]  = trainingSet[i].rating
        userMovies[user].count = count + 1
    end

    return userMovies
end

--Updates the user vector representation based on the updated movie vector representations
function updateUserVectors(config, userMovies, numUsers)
    local k = 0
    for i = 1, numUsers do
        local userMovieInfo = userMovies[i]
        if userMovieInfo.count ~= 0 then
            local userVec = getUser(i, config)
            userVec:zero()

            for j = 1, userMovieInfo.count do
                local rating = userMovieInfo.ratings[j] 
                local movie = userMovieInfo.movieIds[j]
                
                local movieVec = getMovie(movie, config)

                userVec:add(movieVec, (rating - 3))

                k = k + 1
                if k % 10000 == 0 or k == config.trainingSetSize then
                    xlua.progress(k, config.trainingSetSize)
                end
            end
            userVec:div(userMovieInfo.count)
        end
    end
end

--Creates a batch of size config.batch starting at index start
function createBatch(start, dataSet, dataSetSize, config)
    local inputs = torch.CudaTensor(math.min(config.batch, dataSetSize - start + 1), INPUT_VEC_LEN)
    local expected = torch.CudaTensor(math.min(config.batch, dataSetSize - start + 1))

    --Build the batched dataset
    local j = 1
    for j = start, math.min(dataSetSize, start + config.batch - 1) do
        review = dataSet[j] 
        movieVec = getMovie(review.movieId, config)
        userVec = getUser(review.userId, config)

        --Combine the user and movie vectors to create input for the network
        inputs[j - start + 1] = torch.cat(userVec, movieVec, 1):t()
        expected[j - start + 1] = review.rating
    end

    return inputs, expected
end

function train(network, config)
    --Load the two datasets into two tables to avoid extra file reads later
    local trainingSet, trainingSetSize, weights, usersSeen, maxUser = loadDataset(config.trainingSetLocation)
    local validationSet, validationSetSize = loadDataset(config.validationSetLocation, usersSeen)

    config.trainingSetSize = trainingSetSize
    config.validationSetSize = validationSetSize

    local x, dx = network:getParameters()
    local timer = torch.Timer()
    weights = torch.CudaTensor(weights)

    --Define parameters to be used for gradient descent
    local sgdParams = {
        learningRate = config.learningRate,
        momentum = config.momentum,
        weightDecay = config.l2Lambda --This is the L2 regularization parameter
    }

    --Match associate movie reviews to the user that made the review
    local userMovies = userRatingsMatch(trainingSet, trainingSetSize, maxUser)

    --Define the loss function
    local criterion = nn.ClassNLLCriterion(weights):cuda()

    print("[TRAINING] Start Training...")

    --Train on the dataset for the designated number of epochs
    for epoch = config.currentEpoch, config.epochs do
        print(string.format("[TRAINING] Current Epoch: %d", epoch))
        local epochError = {}

        if (epoch % config.learningRateDecay) == 0 then
            print(string.format("[TRAINING] Setting new learning rate to: %f", sgdParams.learningRate / 10))
            sgdParams.learningRate = sgdParams.learningRate / 10
        end

        local averageTrainingError = 0
        local averageTrainingRMSE = 0
        
        shuffle(trainingSet, trainingSetSize)
        timer:reset()
       
        --Update the user vector representations
        updateUserVectors(config, userMovies, maxUser)

        --Enable dropout layers
        network:training()

        --------------------------
        -- TRAINING STARTS HERE --
        --------------------------
        for i = 1, trainingSetSize, config.batch do
            local inputs, expected = createBatch(i, trainingSet, trainingSetSize, config)

            --Define the closure calculation function
            local feval = function(x_new)
                if x ~= x_new then
                    x:copy(x_new)
                end

                --Zero the accumulated gradients
                dx:zero()

                --Perform the training
                local output = network:forward(inputs)
                local loss = criterion:forward(output, expected)
                local gradients = network:backward(inputs, criterion:backward(output, expected))

                local _, outRating = torch.max(output, 2)
                local outRating = outRating:resize(outRating:size(1))

                local trainingError, trainingMSE = getErrors(outRating, expected, trainingSetSize)

                --Keep track of the average error
                averageTrainingError = averageTrainingError + trainingError
                averageTrainingRMSE = averageTrainingRMSE + trainingMSE

                for j = i, math.min(trainingSetSize, i + config.batch - 1) do
                    local review = trainingSet[j]
                    local movieVec = getMovie(review.movieId, config)
                    local movieGrad = gradients[j - i + 1]:narrow(1, USER_VEC_LEN + 1, MOVIE_VEC_LEN)

                    --Apply the updates to the user/movie vectors manually
                    movieVec:add(movieGrad * -sgdParams.learningRate)
                end

                return loss, dx
            end
           
            --Train and update parameters
            local _, fs = optim.sgd(feval, x, sgdParams)
           
            --Print progress within epoch
            if (i % (config.batch * 10)) - 1 == 0 then
                xlua.progress(i, trainingSetSize)
            end
        end
        xlua.progress(trainingSetSize, trainingSetSize)
       
        epochError.trainingError = averageTrainingError
        epochError.trainingRMSE = torch.sqrt(averageTrainingRMSE)
        print(string.format("[TRAINING] Error after epoch %d over %d trials in " ..timer:time().real .. " seconds: %.4f%% RMSE: %.4f", epoch, trainingSetSize, epochError.trainingError, epochError.trainingRMSE))
        
        --Plot the training accuracy curve
        updateTrainingLog(epochError)

        ----------------------------
        -- VALIDATION STARTS HERE --
        ----------------------------
        if validationSetSize > 0 then
            print(string.format("[VALIDATION] Performing validation for epoch %d...", epoch))
            local averageAccuracy = 0
            local averageRMSE = 0
            
            --Disable dropout layers
            network:evaluate()

            for i = 1, validationSetSize, config.batch do
                local inputs, expected = createBatch(i, validationSet, validationSetSize, config)
                local output = network:forward(inputs)
                local _, outRating = torch.max(output, 2)
                outRating = outRating:resize(outRating:size(1))
                
                local errorValue, MSE = getErrors(outRating, expected, validationSetSize)
                averageAccuracy = averageAccuracy + errorValue
                averageRMSE = averageRMSE + MSE

                if (i % (config.batch * 5)) - 1 == 0 then
                    xlua.progress(i, validationSetSize)
                end
            end
            xlua.progress(validationSetSize, validationSetSize)

            epochError.validationError = averageAccuracy
            epochError.validationRMSE = torch.sqrt(averageRMSE)
            
            print(string.format("[VALIDATION] Error after epoch %d over %d trials: %.4f%% RMSE: %.4f", epoch, validationSetSize, epochError.validationError, epochError.validationRMSE))
        
            --Plot the validation accuracy information
            updateValidationLog(epochError)
        end

        ------------------
        -- END OF EPOCH --
        ------------------
        table.insert(config.errors, epochError)
        config.currentEpoch = epoch

        --Save the network after 5 epochs or when we're done training
        if (epoch % 5 == 0) or (epoch == config.epochs)  then
            saveNetwork(network, config, epoch)
        end
    end
    print("[TRAINING] Training Complete!")
end

function getErrors(actual, expected, setSize)
    --Combine the user and movie vectors to create input for the network
    
    local errorValue = torch.sum((torch.abs(actual - expected) * 100 / 5) / setSize)
    local MSE = torch.sum(torch.pow(actual - expected, 2) / setSize)

    return errorValue, MSE
end

function saveNetwork(network, config, epoch)
    --If the user specified a save location, save the network and other config information
    if config.saveLocation ~= nil then
        networkPackage = {
            network = network,
            config = config
        }

        print('Saving trained network at: ' .. config.saveLocation .. epoch .. ".network")
        torch.save(config.saveLocation .. epoch .. ".network", networkPackage)
    end
end

function parseArgs()
    local argparse = require "argparse"
    local parser = argparse()
    parser:option "-e" "--epoch"
          :args(1)
          :description "Number of epochs to train."
          :default "1"
    parser:option "-l" "--load"
          :args(1)
          :description "Load a previously trained network."
    parser:option "-s" "--save"
          :args(1)
          :description "Save the network after training to the given location."
    parser:option "-r" "--resume"
          :description "Resume training a network with a previous training configuration. If defined, all other user parameters are ignored except gpu."
          :args(1)
    parser:option "-a" "--alpha"
          :args(1)
          :description "Set the learning rate for training."
          :default "0.01"
    parser:option "-t" "--trainingset"
          :args(1)
          :description "Location of the training set."
    parser:option "-v" "--validationset"
          :args(1)
          :description "Location of the verification set."
    parser:option "-ad" "--alphadecay"
          :args(1)
          :description "Set the learning rate decay rate for training (epoch interval before decay)."
          :default "100"
    parser:option "--gpu"
          :args(1)
          :description "Set the GPU to train on."
          :default "0"
    parser:option "-m" "--momentum"
          :args(1)
          :description "The momentum to be applied during training."
          :default "0.9"
    parser:option "-b" "--batch"
          :args(1)
          :description "The size of the batches to be used during training."
          :default "32"
    parser:option "--lambda"
          :args(1)
          :description "The lambda parameter used for L2-Regularization."
          :default "0.0"
    parser:flag "--train"
          :description "Train the network."
    return parser:parse()
end

function createNet()
    --Build the network structure
    net = nn.Sequential()
    net:add(nn.Linear(INPUT_VEC_LEN, FIRST_LAYER_SIZE))
    net:add(nn.ReLU())
    net:add(nn.Dropout(0.5))
    net:add(nn.Linear(FIRST_LAYER_SIZE, SECOND_LAYER_SIZE))
    net:add(nn.ReLU())
    net:add(nn.Dropout(0.5))
    net:add(nn.Linear(SECOND_LAYER_SIZE, 5))
--    net:add(nn.ReLU())
--    net:add(nn.Dropout(0.5))
--    net:add(nn.Linear(THIRD_LAYER_SIZE, 5))
    net:add(nn.LogSoftMax())

    --Initialize the weights of the network using xavier initialization
    net = require('weight-init')(net, 'xavier')
    net = net:cuda()
    
    return net
end

function resumeTraining(trainingConfig)
    --Reload the error information from training so we can graph the results again when we resume training
    for i = 1, #trainingConfig.errors do
        epochError = trainingConfig.errors[i]
        updateTrainingLog(epochError)
        if trainingConfig.validationSetLocation ~= nil then
            updateValidationLog(epochError)
        end
    end
end

function updateTrainingLog(epochError)
    trainingLogger:add{[LOGGER_TRAINING] = (100 - epochError.trainingError)}
    trainingRMSELogger:add{[LOGGER_TRAINING] = epochError.trainingRMSE}
    trainingLogger:plot()
    trainingRMSELogger:plot()
    trainingLogger:style{[LOGGER_TRAINING] = "-"}
    trainingRMSELogger:style{[LOGGER_TRAINING] = "-"}
end

function updateValidationLog(epochError)
    validationLogger:add{[LOGGER_VALIDATION] = (100 - epochError.validationError)}
    validationRMSELogger:add{[LOGGER_VALIDATION] = epochError.validationRMSE}
    validationLogger:style{[LOGGER_VALIDATION] = "-"}
    validationRMSELogger:style{[LOGGER_VALIDATION] = "-"}
    validationLogger:plot()
    validationRMSELogger:plot()
end

------------------------|
-- START MAIN PROGRAM --|
------------------------|

local args = parseArgs()
trainingLogger = optim.Logger("training.log")
trainingRMSELogger = optim.Logger("trainingRMSE.log")
validationLogger = optim.Logger("validation.log")
validationRMSELogger = optim.Logger("validationRMSE.log")

trainingConfig = {}
net = {}

trainingConfig.saveLocation = args.save
trainingConfig.loadLocation = args.load 
trainingConfig.resumeLocation = args.resume
trainingConfig.epochs = tonumber(args.epoch)
trainingConfig.currentEpoch = 0 
trainingConfig.userVecs = {} 
trainingConfig.movieVecs = {}
trainingConfig.learningRate = tonumber(args.alpha)
trainingConfig.learningRateDecay = tonumber(args.alphadecay)
trainingConfig.momentum = tonumber(args.momentum)
trainingConfig.batch = tonumber(args.batch)
trainingConfig.l2Lambda = tonumber(args.lambda)
trainingConfig.errors = {}

--Load the dataset locations.
trainingConfig.trainingSetLocation = DEFAULT_TRAINING_SET_DIR
if args.trainingset ~= nil then
    trainingConfig.trainingSetLocation = args.trainingset
end

trainingConfig.validationSetLocation = DEFAULT_VALIDATION_SET_DIR
if args.validationset ~= nil then
    trainingConfig.validationSetLocation = args.validationset
end

print(string.format("[TRAINING] Training on set: %s", trainingConfig.trainingSetLocation))
print(string.format("[VALIDATION] Validating with set: %s", trainingConfig.validationSetLocation))

--Set the GPU to train on.
cutorch.setDevice(tonumber(args.gpu) + 1)

--Check to see if we want to reload a previously trained network or we want to resume training on a network
if trainingConfig.loadLocation ~= nil or trainingConfig.resumeLocation ~= nil then
    --Load a previously trained network
    location = trainingConfig.resumeLocation or trainingConfig.loadLocation

    print("Loading previously trained network from: '" .. location .. "'...")
    savedPackage = torch.load(location)
    net = savedPackage.network

    --If resuming, just reload the old config to pick up where we left off
    if trainingConfig.resumeLocation ~= nil then
        resumeTraining(savedPackage.config)
    else
        --If we're just loading a network, we just need to reload the user/movie vectors
        trainingConfig.userVecs = savedPackage.config.userVecs
        trainingConfig.movieVecs = savedPackage.config.movieVecs
    end

    print('Network:\n' .. net:__tostring())
    print("Loading complete!")
else
    --Otherwise, train a new network
    print("No previously trained network specified. Training a new network.")
    print("Generating network structure...")
    net = createNet()
    print("Network structure creation complete.")
    print('Network:\n' .. net:__tostring())
end

if args.train then
    --Increment the epoch counter just in case we're resuming training on a network
    trainingConfig.currentEpoch = trainingConfig.currentEpoch + 1
    train(net, trainingConfig)
    printCSV(config)
end

