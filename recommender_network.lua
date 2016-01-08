------------------|
--PACKAGE IMPORTS |
------------------|
require 'nn'
require 'optim'
require 'cutorch'
require 'cunn'

------------------------------|
-- START CONSTANTS DEFINITION |
------------------------------|
DEFAULT_TRAINING_SET_DIR = './dataset/training_set/training/'
DEFAULT_VALIDATION_SET_DIR = './dataset/training_set/validation/'

USER_VEC_LEN  = 16 
MOVIE_VEC_LEN = 48
INPUT_VEC_LEN = USER_VEC_LEN + MOVIE_VEC_LEN
FIRST_LAYER_SIZE  = 128 
SECOND_LAYER_SIZE = 192
THIRD_LAYER_SIZE  = 256

LOGGER_TRAINING = "training_data"
LOGGER_VALIDATION = "validation_data"

MOVIE_ID_INDEX = 1
USER_ID_INDEX = 2
RATING_INDEX = 3

------------------------------|
-- END CONSTANT DEFINITIONS   |
------------------------------|

--Returns the names of all files in a given directory using find
function getFileNames(directory)
    local i, t, popen = 0, {}, io.popen
    for filename in popen('find "'..directory..'" -maxdepth 1 -type f -printf "%f\n"'):lines() do
        i = i + 1
        t[i] = filename
    end
    return t
end

--Standard string split method
function split(string, delim)
    if delim == nil then
        delim = "%s"
    end
    local t = {}
    local i = 1
    for str in string.gmatch(string, "([^"..delim.."]+)") do
        t[i] = str
        i = i + 1
    end

    return t
end

--Parse the raw review and put its information into a table
function createReview(rawReview)
    local splitReview = split(rawReview, ',')
    local review = torch.Tensor(3)
    review[USER_ID_INDEX] = tonumber(splitReview[1])
    review[RATING_INDEX]  = tonumber(splitReview[2])
    --[[local review = {
        userId = tonumber(splitReview[1]),
        rating = tonumber(splitReview[2]),
        --date   = splitReview[3]
    }]]--
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

--Load the movie rating information from the given directory into a table
--of movie tables which contain the movie ID as well as reviews associated
--with the movie
function loadDataset(directory)
    if directory == nil then return {} end

    local files = getFileNames(directory)
    local dataset = {}
    local reviews = {}
    
    for i = 1, #files do
        if i % 100 == 0 then 
            collectgarbage()
        end
        local f = io.open(directory..files[i], 'r')
        local movieString = f:read()
        local movieId = tonumber(string.sub(movieString, 0, #movieString - 1))

        while true do
            local review = getReview(f:read())
            if review == nil then break end
            --review.movieId = movieId
            review[MOVIE_ID_INDEX] = movieId
            table.insert(reviews, review)
        end
        f:close()
    end

    return reviews 
end

--Randomly shuffles the dataset for better training results
function shuffleDataset(dataset)
    local rand = math.random
    assert(dataset, "shuffleDataset() expected a table, got nil")
    local iterations = #dataset
    local j

    for i = iterations, 2, -1 do
        j = rand(i)
        dataset[i], dataset[j] = dataset[j], dataset[i]
    end
end

function train(network, config)
    --Load the two datasets into two tables to avoid extra file reads later
    local trainingSet = loadDataset(config.trainingSetLocation)
    local validationSet = loadDataset(config.validationSetLocation)
    local errors = {}

    local x, dx = network:getParameters()

    local timer = torch.Timer()

    local weights = torch.CudaTensor(5)
    --Define parameters to be used for gradient descent
    sgdParams = {
        learningRate = config.learningRate,
        momentum = config.momentum,
        weightDecay = config.l2Lambda --This is the L2 regularization parameter
    }

    local usersSeen = {}

    for i = 1, 5 do
        weights[i] = 0
    end

    for i = 1, #trainingSet do
        usersSeen[trainingSet[i][USER_ID_INDEX]] = true
        weights[trainingSet[i][RATING_INDEX]] = weights[trainingSet[i][RATING_INDEX]] + (1 / #trainingSet)
    end

    --Use RMSE Error as loss function
    local criterion = nn.AbsMultiMarginCriterion(2, weights):cuda()

    for i = #validationSet, 1, -1 do
        if not usersSeen[validationSet[i][USER_ID_INDEX]] then
            table.remove(validationSet, i)
        end
    end

    print("[TRAINING] Start Training...")
    --Train on the dataset for the designated number of epochs
    for epoch = 1, config.epoch do
        print(string.format("[TRAINING] Current Epoch: %d", epoch))
        movieCount = 0
        epochError = {}

        if (epoch % config.learningRateDecay) == 0 then
            print(string.format("[TRAINING] Setting new learning rate to: %f", sgdParams.learningRate / 10))
            sgdParams.learningRate = sgdParams.learningRate / 10
        end

        local averageTrainingError = 0
        local averageTrainingRMSE = 0
        shuffleDataset(trainingSet)

        timer:reset()
        
        --Enable dropout layers
        network:training()

        for i = 1, #trainingSet, config.batch do
            --Force garbage collection every once in awhile
            --Lua has a max memory limit of 2GB, and apparently the garbage collector doesn't like to run
            --inside of loops. On top of that, these local variables are probably causing additional memory writes
            --which is bringing us over the 2GB limit (ran into these issues with a subset of the Netflix Prize dataset)
            if i % 25000 == 0 then
                collectgarbage()
            end

            local inputs = torch.CudaTensor(math.min(config.batch, #trainingSet - i + 1), INPUT_VEC_LEN)
            local expected = torch.CudaTensor(math.min(config.batch, #trainingSet - i + 1))

 
            --Build the batched dataset
            j = 1
            for j = i, math.min(#trainingSet, i + config.batch - 1) do
                local review = trainingSet[j] 
                local movieVec = getMovie(review[MOVIE_ID_INDEX], config)
                local userVec = getUser(review[USER_ID_INDEX], config)

                --Combine the user and movie vectors to create input for the network
                inputs[j - i + 1] = torch.cat(userVec, movieVec, 1):t()
                expected[j - i + 1] = review[RATING_INDEX]
            end
            
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
                    outRating = outRating:resize(outRating:size(1))

                    --Keep track of the average error
                    averageTrainingError = averageTrainingError + torch.sum(((torch.abs(outRating - expected) * 100) / 5)/#trainingSet)

                    --print(string.format("%f %d", averageTrainingError, i))
                    averageTrainingRMSE = averageTrainingRMSE + torch.sum(torch.pow(outRating - expected, 2) / #trainingSet)

                    for j = i, math.min(#trainingSet, i + config.batch - 1) do
                        local review = trainingSet[j]
                        local userVec = getUser(review[USER_ID_INDEX], config)
                        local movieVec = getMovie(review[MOVIE_ID_INDEX], config)

                        local userGrad = gradients[j - i + 1]:narrow(1, 1, USER_VEC_LEN)
                        local movieGrad = gradients[j - i + 1]:narrow(1, USER_VEC_LEN + 1, MOVIE_VEC_LEN)

                        --Apply the updates to the user/movie vectors manually
                        userVec:add(userGrad * -sgdParams.learningRate)
                        movieVec:add(movieGrad * -sgdParams.learningRate)
                    end

                    return loss, dx
            end
           
            --Train and update parameters
            _, fs = optim.sgd(feval, x, sgdParams)
            ---network:syncParameters()
            --Check to see if we are also training the user/movie vectors
            --Print progress within epoch
            xlua.progress(i, #trainingSet)
        end
       
        --print(averageTrainingError)
        epochError.trainingError = averageTrainingError
        epochError.trainingRMSE = torch.sqrt(averageTrainingRMSE)
        print(string.format("[TRAINING] Error after epoch %d over %d trials in " ..timer:time().real .. " seconds: %.4f%% RMSE: %.4f", epoch, #trainingSet, epochError.trainingError, epochError.trainingRMSE))
        
        --Plot the training accuracy curve
        trainingLogger:add{[LOGGER_TRAINING] = (100 - epochError.trainingError)}
        trainingRMSELogger:add{[LOGGER_TRAINING] = epochError.trainingRMSE}
        trainingLogger:style{[LOGGER_TRAINING] = "-"}
        trainingRMSELogger:style{[LOGGER_TRAINING] = "-"}
        trainingLogger:plot()
        trainingRMSELogger:plot()
        
        if #validationSet > 0 then
            print(string.format("[VALIDATION] Performing validation for epoch %d...", epoch))
            local averageAccuracy = 0.0
            local averageRMSE = 0
            local averageAccuracyCount = 0
            
            --Disable dropout layers
            network:evaluate()

            for i = 1, #validationSet, config.batch do
                --Force garbage collection every once in awhile
                --Lua has a max memory limit of 2GB, and apparently the garbage collector doesn't like to run
                --inside of loops. On top of that, these local variables are probably causing additional memory writes
                --which is bringing us over the 2GB limit (ran into these issues with a subset of the Netflix Prize dataset)
                if i % 25000 == 0 then
                    collectgarbage()
                end

                local inputs = torch.CudaTensor(math.min(config.batch, #validationSet - i + 1), INPUT_VEC_LEN)
                local expected = torch.CudaTensor(math.min(config.batch, #validationSet - i + 1), 1)

                j = 1
                for j = i, math.min(#validationSet, i + config.batch - 1) do
                    local review = validationSet[j] 
                    local movieVec = getMovie(review[MOVIE_ID_INDEX], config)
                    local userVec = getUser(review[USER_ID_INDEX], config)

                    --Combine the user and movie vectors to create input for the network
                    inputs[j - i + 1] = torch.cat(userVec, movieVec, 1):t()
                    expected[j - i + 1] = review[RATING_INDEX]
                end

                local errorValue, RMSE = predictRating(network, config, inputs, expected, #validationSet)

                averageAccuracy = averageAccuracy + errorValue
                averageRMSE = averageRMSE + RMSE
            end
            epochError.validationError = averageAccuracy
            epochError.validationRMSE = torch.sqrt(averageRMSE)
            
            print(string.format("[VALIDATION] Error after epoch %d over %d trials: %.4f%% RMSE: %.4f", epoch, #validationSet, epochError.validationError, epochError.validationRMSE))

            --Plot the validation acccuracy curve
            validationLogger:add{[LOGGER_VALIDATION] = (100 - epochError.validationError)}
            validationRMSELogger:add{[LOGGER_VALIDATION] = epochError.validationRMSE}
            validationLogger:style{[LOGGER_VALIDATION] = "-"}
            validationRMSELogger:style{[LOGGER_VALIDATION] = "-"}
            validationLogger:plot()
            validationRMSELogger:plot()
        end

        --Save the network after 20 epochs or when we're done training
        if (epoch % 5 == 0) or (epoch == config.epoch)  then
            saveNetwork(network, config, epoch)
        end

        table.insert(errors, epochError)
   end

    print("[TRAINING] Training Complete!")

    print(string.format("Epoch, Training Error, Validation Error, Training RMSE, Validation RMSE"))
    for i = 1, #errors do
        epochError = errors[i]
        if epochError.validationError == nil then
            epochError.validationError = 0
            epochError.validationRMSE = 0
        end
        print(string.format("%d, %.4f%%, %.4f%%, %.4f, %.4f", i, epochError.trainingError, epochError.validationError, epochError.trainingRMSE, epochError.validationRMSE))
    end 
end

function predictRating(network, config, inputs, rating, setSize)
    --Combine the user and movie vectors to create input for the network
    local output = network:forward(inputs)
    local _, outRating = torch.max(output, 2)
    outRating = outRating:resize(outRating:size(1))
    
    local errorValue = torch.sum((torch.abs(outRating - rating) * 100 / 5) / setSize)
    local RMSE = torch.sum(torch.pow(outRating - rating, 2) / setSize)

    return errorValue, RMSE
end

function saveNetwork(network, config, epoch)
    --If the user specified a save location, save the network and other config information
    if saveLocation ~= nil then
        networkPackage = {
            network = network,
            config = config
        }

        print('Saving trained network at: ' .. saveLocation .. epoch .. ".network")
        torch.save(saveLocation .. epoch .. ".network", networkPackage)
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


------------------------|
-- START MAIN PROGRAM --|
------------------------|

local args = parseArgs()
local loadLocation = args.load
saveLocation = args.save
trainingLogger = optim.Logger("training.log")
trainingRMSELogger = optim.Logger("trainingRMSE.log")
validationLogger = optim.Logger("validation.log")
validationRMSELogger = optim.Logger("validationRMSE.log")

trainingConfig = {}
net = {}

trainingConfig.epoch = tonumber(args.epoch)
trainingConfig.userVecs = {} 
trainingConfig.movieVecs = {}
trainingConfig.learningRate = tonumber(args.alpha)
trainingConfig.learningRateDecay = tonumber(args.alphadecay)
trainingConfig.momentum = tonumber(args.momentum)
trainingConfig.batch = tonumber(args.batch)
trainingConfig.l2Lambda = tonumber(args.lambda)

cutorch.setDevice(tonumber(args.gpu) + 1)

if loadLocation ~= nil then
    --Load a previously trained network
    print("Loading previously trained network from: '" .. loadLocation .. "'...")
    savedPackage = torch.load(loadLocation)
    net = savedPackage.network
    trainingConfig.userVecs = savedPackage.config.userVecs
    trainingConfig.movieVecs = savedPackage.config.movieVecs
    print('TestNet:\n' .. net:__tostring())

    --TODO Reload training config?

    print("Loading complete!")
else
    --Otherwise, train a new network
    print("No previously trained network specified. Training a new network.")
    print("Generating network structure...")
    --Build the network structure
    net = nn.Sequential()
    
    net:add(nn.Linear(INPUT_VEC_LEN, FIRST_LAYER_SIZE))
    net:add(nn.ReLU())
--    net:add(nn.Dropout(0.5))
    net:add(nn.Linear(FIRST_LAYER_SIZE, 5))
--    net:add(nn.ReLU())
--    net:add(nn.Linear(SECOND_LAYER_SIZE, 5))
--    net:add(nn.ReLU())
--    net:add(nn.Linear(THIRD_LAYER_SIZE, 5))
    net:add(nn.LogSoftMax())

    --Initialize the weights of the network using xavier initialization
    net = require('weight-init')(net, 'xavier')
    
    --[[parallel_net = nn.DataParallel(1):cuda()
    for i = 1, cutorch.getDeviceCount() do
        cutorch.setDevice(i)
        parallel_net:add(net:clone():cuda(), i)
    end
    cutorch.setDevice(1)
    net = parallel_net]]--
    net = net:cuda()
    print("Network structure creation complete.")
    print('TestNet:\n' .. net:__tostring())
end

if args.train then
    trainingConfig.trainingSetLocation = DEFAULT_TRAINING_SET_DIR
    if args.trainingset ~= nil then
        trainingConfig.trainingSetLocation = args.trainingset
    end
    print(string.format("[TRAINING] Training on set: %s", trainingConfig.trainingSetLocation))

    trainingConfig.validationSetLocation = nil
    if args.validationset ~= nil then
        trainingConfig.validationSetLocation = args.validationset
        print(string.format("[VALIDATION] Validating with set: %s", args.validationset))
    end

    train(net, trainingConfig) 
end

