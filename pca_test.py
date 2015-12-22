#!/usr/bin/python
#
# Created by Albert Au Yeung (2010)
#
# An implementation of matrix factorization
#
import os
import math
try:
    import numpy
except:
    print "This implementation requires the numpy module."
    exit(0)

def load_reviews(path):
    maxUser = 0
    maxMovie = 0
    users = []

    for fileName in os.listdir(path):
        if os.path.isdir(path + fileName):
            continue

        with open(path + fileName, 'r') as f:
            movieId = int(f.readline()[:-2]) # Remove colon
            if movieId > maxMovie:
                maxMovie = movieId

            for line in f:
                splitReview = line.split(',')
                userId = int(splitReview[0])
                users.append(userId)

                if userId > maxUser:
                    maxUser = userId
    
    matrix = []
    #for i in range(maxUser):
    #    row = []
    #    for j in range(maxMovie):
    #        row.append(0.0)
    #    matrix.append(row)

    s = (maxUser, maxMovie)
    matrix = numpy.zeros(s)

    for fileName in os.listdir(path):
        if os.path.isdir(path + fileName):
            continue

        with open(path + fileName, 'r') as f:
            movieId = int(f.readline()[:-2]) # Remove colon
            for line in f:
                splitReview = line.split(',')
                userId = int(splitReview[0])
                rating = float(splitReview[1])
                
                matrix[userId - 1][movieId - 1] = rating
    return matrix, users 

def load_validation(path):
    reviews = []
    for fileName in os.listdir(path):
        if os.path.isdir(path + fileName):
            continue

        with open(path + fileName, 'r') as f:
            movieId = int(f.readline()[:-2]) # Remove colon
            for line in f:
                splitReview = line.split(',')
                review = {}
                review["movie"] = movieId
                review["user"] = int(splitReview[0])
                review["rating"] = int(splitReview[1])
                reviews.append(review)
    return reviews



###############################################################################

"""
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
"""
def matrix_factorization(R, P, Q, K, steps=20, alpha=0.0002, beta=0.02):
    Q = Q.T
    for step in xrange(steps):
        print "Epoch: " + str(step + 1) + "/" + str(steps)
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - numpy.dot(P[i,:],Q[:,j])
                    for k in xrange(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        print "Done prediction phase."
        eR = numpy.dot(P,Q)
        e = 0
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
                    for k in xrange(K):
                        e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
        print "Done training phase."
        if e < 0.001:
            break
    return P, Q.T

###############################################################################

if __name__ == "__main__":
    R = [
         [5,3,0,1],
         [4,0,0,1],
         [1,1,0,5],
         [1,0,0,4],
         [0,1,5,4],
        ]
    validation = load_validation("./dataset/training_set/training_set_test/validation/")

    R, users = load_reviews("./dataset/training_set/training_set_test/testing/")
    R = numpy.array(R)

    N = len(R)
    M = len(R[0])
    K = 900 

    P = numpy.random.rand(N,K)
    Q = numpy.random.rand(M,K)

    nP, nQ = matrix_factorization(R, P, Q, K)
    nR = numpy.dot(nP, nQ.T)
    print(nR)

    validationError = 0
    validationRMSE = 0
    validationCount = 0

    for review in validation:
        userId = review["user"]
        movieId = review["movie"]
        rating = review["rating"]

        if not userId in users:
            continue

        validationCount = validationCount + 1
        predictedRating = round(nR[userId-1][movieId-1])

        validationError = validationError + (abs(rating - predictedRating) / 5)
        validationRMSE = validationRMSE + pow(rating - predictedRating, 2)

    print "Validation Count: " + str(validationCount)
    print "Validation Accuracy: " + str((1 - (validationError / validationCount)) * 100)
    print "Validation RMSE: " + str(math.sqrt(validationRMSE / validationCount))

