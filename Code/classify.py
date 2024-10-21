import pandas as pd
import csv

def test(inFile, inSpam, inHam, outFile):
    #Open test file for reading
    myfile = open(inFile, "r")
    myline = myfile.readline()

    #Read file data, remove ',,,' and the new line character and add to rows
    rows = []
    while myline:
        myline = myfile.readline()
        rowQ = myline.strip(',,,\n')
        row = rowQ.split(",", 1)
        rows.append(row)
    #Remove the last element since its empty
    rows = rows[:-1]
    myfile.close()

    #Convert the rows into a pandas DataFrame with columns Label and SMS
    data = pd.DataFrame(rows, columns=['Label', 'SMS'])

    #Remove punctuation and make SMS lower cased
    data['SMS'] = data['SMS'].str.replace('\W', ' ', regex=True) # Removes punctuation
    data['SMS'] = data['SMS'].str.lower()

    #Open Spam probability file for reading
    spamFile = open(inSpam, "r")
    spamreader = csv.reader(spamFile)
    #Read the header
    spamHead = next(spamreader)
    #From the header, get the total number of spam words, and the prior spam probability
    n_spam = int(spamHead[0])
    p_spam = float(spamHead[1])

    #Read data from spam probability file and add to spamRows
    spamRows = []
    for row in spamreader:
        spamRows.append(row)
    spamFile.close()

    #Open Ham probability file for reading
    hamFile = open(inHam, "r")
    hamreader = csv.reader(hamFile)
    #Read the header
    hamHead = next(hamreader)
    #From the header, get the total number of ham words, and the prior ham probability
    n_ham = int(hamHead[0])
    p_ham = float(hamHead[1])

    #Read data from ham probability file and add to hamRows
    hamRows = []
    for row in hamreader:
        hamRows.append(row)
    hamFile.close()

    #Get a list of uniqye words and assign it to vocabulary
    vocabulary = [vRow[0] for vRow in spamRows]
    #Number of unique words
    n_vocabulary = len(vocabulary)

    #Laplace smoothing, Alpha = 1
    alpha = 1

    #Create a list of spam/ham probabilities based on a unique word from the vocabulary and set all to 0
    parameters_spam = {unique_word:0 for unique_word in vocabulary}
    parameters_ham = {unique_word:0 for unique_word in vocabulary}

    #Iterate through the words from vocabulary
    i = 0
    for word in vocabulary:
        #Get the number of instances for certain word occurs in spam messages
        n_word_given_spam = int(spamRows[i][1])
        #Calculate P(word|spam)
        p_word_given_spam = (n_word_given_spam + alpha) / (n_spam + alpha*n_vocabulary)
        #Add it to the list of spam probabilities
        parameters_spam[word] = p_word_given_spam

        #Get the number of instances a certain word occurs in ham messages
        n_word_given_ham = int(hamRows[i][1])
        #Calculate P(word|ham)
        p_word_given_ham = (n_word_given_ham + alpha) / (n_ham + alpha*n_vocabulary)
        #Add it to the list of ham probabilities
        parameters_ham[word] = p_word_given_ham
        
        i += 1

    #Nested function that classifies whether a message is ham or spam
    def classify_test_set(message):
        #Assigning p_spam and p_ham to these variables
        p_spam_given_message = p_spam
        p_ham_given_message = p_ham

        #Iterates through each word in a message
        for word in message:
            #Check if word exists in the spam list of words
            if word in parameters_spam:
                #Multiply P(word|spam) with what used to be P(spam) and assign it to itself
                p_spam_given_message *= parameters_spam[word]

            #Check if word exists in the ham list of words
            if word in parameters_ham:
                #Multiply P(word|ham) with what used to be P(ham) and assign it to itself
                p_ham_given_message *= parameters_ham[word]

        #From the final product of these calculation, assign ham/spam based on which is greater
        if p_ham_given_message > p_spam_given_message:
            return 'ham'
        elif p_spam_given_message > p_ham_given_message:
            return 'spam'
        #If the two are same, then return spam because it increases the classification accuracy
        else:
            return 'spam'

    #Apply the classification funtion to the test data and create a predicted column
    data['predicted'] = data['SMS'].apply(classify_test_set)

    #Open the output prediction file 
    predFile = open(outFile, "w")
    predictions = data['predicted']
    #Iterate through all the preuctions and write them to the output file
    for j in predictions:
        predString = j + "\n"
        predFile.write(predString)
    predFile.close()

    #Iterate through data and check if the prediction matches the label and calculate classification accuracy
    correct = 0
    total = data.shape[0]
    for pred in data.iterrows():
        pred = pred[1]
        if pred['Label'] == pred['predicted']:
            correct += 1

    #On average return an accuracy of about 86-88%
    print('Correct:', correct)
    print('Incorrect:', total - correct)
    print('Accuracy:', correct/total)