# This is a function for k-fold cross-validation on (X; y)
# Yi Ding

# This function return the accuracy score of the prediction
def my_accuracy_score(ytrue,ypred):
    import numpy as np    
    ytrue = np.array(ytrue)
    ypred = np.array(ypred)
    if ytrue.shape[0] != ypred.shape[0]:
        print("ERROR: ytrue and ypred not same length!")
        return
    accuracy_score = 0;
    for i in range(0,ytrue.shape[0]):
        if ytrue[i] == ypred[i]:
            accuracy_score = accuracy_score + 1;
    return (float(accuracy_score)/float(ytrue.shape[0]))
	
# Main function
def my_cross_val(method,X,y,k):    
    import numpy as np
    X = np.array(X)
    y = np.array(y)
    y = np.reshape(y,(X.shape[0],1))    
    # Initialize array for the test set error    
    errRat = np.empty([k, 1])
    # Permute the indices randomly
    rndInd = np.random.permutation(y.size)
    # Start and end index of test set
    sttInd = 0;
    endInd = (np.array(y.size/k).astype(int))
    indLen = (np.array(y.size/k).astype(int))
    for i in range(0, k):
        # Prepare training data and test data
        Xtrain = np.concatenate((X[rndInd[0:sttInd],:],X[rndInd[endInd:y.size],:]), axis=0)
        ytrain = np.concatenate((y[rndInd[0:sttInd]],y[rndInd[endInd:y.size]]), axis=0)
        Xtest = X[rndInd[sttInd:endInd],:]
        ytest = y[rndInd[sttInd:endInd]]
        sttInd = endInd
        endInd = endInd + indLen
        # Create the model
        myMethod = method()    
        # Fit the data
        myMethod.fit(Xtrain,ytrain.ravel())
        # Test the model on (new) data
        ypred = myMethod.predict(Xtest)
        # Save error rate
        errRat[i] = 1 - my_accuracy_score(ytest, ypred)
    return errRat