import pandas as pd

def bayes(inFile, outSpam, outHam):
    #Open file for reading
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
    data['SMS'] = data['SMS'].str.replace('\W', ' ', regex=True)
    data['SMS'] = data['SMS'].str.lower()

    #Split all the SMS messages into separate words
    data['SMS'] = data['SMS'].str.split()

    #Create a list of unique words called vocabulary
    vocabulary = []
    for sms in data['SMS']:
        for word in sms:
            vocabulary.append(word)
    vocabulary = list(set(vocabulary))

    #For each word in vocabulary, count the number of times it exists in each message
    word_counts_per_sms = {unique_word: [0] * len(data['SMS']) for unique_word in vocabulary}
    for index, sms in enumerate(data['SMS']):
        for word in sms:
            word_counts_per_sms[word][index] += 1

    #Create a dataframe out of the words and their respective word counts    
    word_counts = pd.DataFrame(word_counts_per_sms)
    
    #Concatenate the original dataframe with the word counts dataframe
    data_join = pd.concat([data, word_counts], axis=1)
    
    #Separate the new dataframe into spam and ham dataframes
    spam_messages = data_join[data_join['Label'] == 'spam']
    ham_messages = data_join[data_join['Label'] == 'ham']

    #Calculate prior probabilities of spam and ham
    p_spam = len(spam_messages) / len(data_join)
    p_ham = len(ham_messages) / len(data_join)

    #Calculate the total number of spam words
    n_words_per_spam_message = spam_messages['SMS'].apply(len)
    n_spam = n_words_per_spam_message.sum()

    #Calculate the total number of ham words
    n_words_per_ham_message = ham_messages['SMS'].apply(len)
    n_ham = n_words_per_ham_message.sum()

    #Open the spam and ham probability files
    spamFile = open(outSpam, "w")
    hamFile = open(outHam, "w")
    
    #Write the total number of spam words and the prior spam probability as the header
    spamHead = str(n_spam) + "," + str(p_spam) + "\n"
    spamFile.write(spamHead)

    #Write the total number of ham words and the prior ham probability as the header
    hamHead = str(n_ham) + "," + str(p_ham) + "\n"
    hamFile.write(hamHead)

    #Traverse through each unique word in the vocabulary
    for word in vocabulary:
        #Calculate the number of times a word exists in the spam messages
        n_word_given_spam = spam_messages[word].sum()
        #Write the word and its respective word count to the spam probability file
        sWordCount = word + "," + str(n_word_given_spam) + "\n"
        spamFile.write(sWordCount)

        #Calculate the number of times a word exists in the ham messages
        n_word_given_ham = ham_messages[word].sum()
        #Write the word and its respective word count to the ham probability file
        hWordCount = word + "," + str(n_word_given_ham) + "\n"
        hamFile.write(hWordCount)