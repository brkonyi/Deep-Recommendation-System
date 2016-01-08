require 'nn' 
require 'cutorch'

MOVIE_ID_INDEX = 1
USER_ID_INDEX = 2
RATING_INDEX = 3

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
    local review = {} 
    review[USER_ID_INDEX] = tonumber(splitReview[1])
    review[RATING_INDEX]  = tonumber(splitReview[2])
    --Note: movie ID is populated in loadDataset, so no, it wasn't forgotten!
    return review
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
    print("Loading dataset...")
    if directory == nil then return {} end

    local files = getFileNames(directory)
    local dataset = {}
    local users = {}
    local maxMovie = 0
    local maxUser = 0

    for i = 1, #files do
        if i % 100 == 0 then 
            collectgarbage()
        end
        local f = io.open(directory..files[i], 'r')
        local movieString = f:read()
        local movieId = tonumber(string.sub(movieString, 0, #movieString - 1))

        if movieId > maxMovie then
            maxMovie = movieId
        end

        while true do
            local review = getReview(f:read())
            if review == nil then break end
            
            if review[USER_ID_INDEX] > maxUser then
                maxUser = review[USER_ID_INDEX]
            end
        end
        f:close()
    end
   
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
            review[MOVIE_ID_INDEX] = movieId
            table.insert(users, review[USER_ID_INDEX])
            table.insert(dataset, review)
        end
        f:close()
    end
    ratingsMatrix = torch.Tensor(dataset)
    print("Dataset loaded.")
    return ratingsMatrix, users, maxUser, maxMovie 
end

function loadValidation(directory, users)
    print("Loading dataset...")
    if directory == nil then return {} end

    local files = getFileNames(directory)
    local dataset = {}

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
            review[MOVIE_ID_INDEX] = movieId

            if users[review[USER_ID_INDEX]] ~= nil then
                table.insert(dataset, review)
            end
        end
        f:close()
    end

    dataset = torch.Tensor(dataset)
    return dataset 
end

--[[
@INPUT:
    R     : a matrix to be factorized, dimension N x M
    P     : an initial matrix of dimension N x K
    Q     : an initial matrix of dimension M x K
    K     : the number of latent features
    steps : the maximum number of steps to perform the optimisation
    alpha : the learning rate
    beta  : the regularization parameter
@OUTPUT:
    the final matrices P and Q
]]--
function matrix_factorization(R, P, Q, K, steps, alpha, beta)
    Q = Q:t()
    for step = 1, steps do
        print(string.format("Epoch: %d/%d", step, steps))
        for i = 1, R:size(1) do
            
            local rating = R[i][RATING_INDEX]
            local movieId = R[i][MOVIE_ID_INDEX]
            local userId = R[i][USER_ID_INDEX]
            
            --TODO this could probably be vectorized a bit better
            local err = rating - torch.dot(P:select(1, userId), Q:select(2, movieId))
            for k = 1, K do
                P[userId][k] = P[userId][k] + alpha * (2 * err * Q[k][movieId] - beta * P[userId][k])
                Q[k][movieId] = Q[k][movieId] + alpha * (2 * err * P[userId][k] - beta * Q[k][movieId])
            end
        end

        print "Done prediction phase."
        
        e = 0
        for i = 1, R:size(1) do
            local rating = R[i][RATING_INDEX]
            local movieId = R[i][MOVIE_ID_INDEX]
            local userId = R[i][USER_ID_INDEX]
            e = e + torch.pow(rating - torch.dot(P:select(1, userId), Q:select(2, movieId)), 2)
            for k = 1, K do
                e = e + (beta/2) * ( torch.pow(P[userId][k],2) + torch.pow(Q[k][movieId],2) )
            end
        end
        
        print "Done training phase."
        if e < 0.001 then
            break
        end
    end
    return P, Q:t()
end


R, users, N, M = loadDataset("./dataset/training_set/training/")
R = R:cuda()

validation = loadValidation("./dataset/training_set/validation/", users):cuda()
K = 2 

P = torch.rand(N,K):cuda()
Q = torch.rand(M,K):cuda()

nP, nQ = matrix_factorization(R, P, Q, K, 1, 0.0002, 0.02)
nQ = nQ:t()
validationError = 0
validationRMSE = 0
validationCount = 0

for i = 1, validation:size(1) do
    review = validation[i]
    userId = review[USER_ID_INDEX]
    movieId = review[MOVIE_ID_INDEX]
    rating = review[RATING_INDEX]

    validationCount = validationCount + 1
    predictedRating = torch.round(torch.dot(nP:select(1, userId), nQ:select(2, movieId)))

    validationError = validationError + (torch.abs(rating - predictedRating) / 5)
    validationRMSE = validationRMSE + torch.pow(rating - predictedRating, 2)
end

print(string.format("Validation Count: %d", validationCount))
print(string.format("Validation Accuracy: %.4f", ((1 - (validationError / validationCount)) * 100)))
print(string.format("Validation RMSE: %.4f", math.sqrt(validationRMSE / validationCount)))
