from os import listdir
from argparse import ArgumentParser
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.feature_extraction.text  import TfidfTransformer
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_regression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import numpy as np
import sklearn
import argparse
import matplotlib.pyplot as plt
from argparse import ArgumentParser

#Checks whether the arguments have been entered correctly.
def Process_args():
    isValid = True
    parser = argparse.ArgumentParser(description='Perform machine learning on the bbc articles set.')
    parser.add_argument('--Fs', metavar='The maximum number of features after reduction', type=int, nargs=1, 
    help='What size the feature set should be reduced to.', default=[600])
    parser.add_argument('--Vs', metavar='Size of the vocabulary to use.', type=int, nargs=1, 
    help='What size the feature set should be reduced to.', default=[1000])
    parser.add_argument('--Splits', metavar='Train, Val, Test splits.', type=int, nargs=3, 
    help='What size the feature set should be reduced to.', default=[80, 10, 10])
    parser.add_argument('--Seed', metavar='Seed for randomness', type=int, nargs=1, 
    help='The seed used for randomness in the program', default=[1337])
    args = vars(parser.parse_args())
    print()
    print("Arguments being used: ", args)
    FEATURE_SIZE = args["Fs"][0]
    VOCAB_SIZE = args["Vs"][0]
    SEED = args["Seed"][0]
    SPLITS = args["Splits"]
    if FEATURE_SIZE >= VOCAB_SIZE + 7:
        print()
        print("--Fs must be lower than 7 + value of --Vs. (Default 1000)")
        isValid = False
    if sum(SPLITS) != 100:
        print()
        print("--Splits must add up to 100. (Default 80, 10, 10)")
        isValid = False
    if not all(i > 0 for i in SPLITS):
        print()
        print("--Splits must all be greater than 0. (Default 80, 10, 10)")
        isValid = False
    return isValid, args, FEATURE_SIZE, VOCAB_SIZE, SPLITS, SEED

#Imports files from the relative bbc folder.
def import_files():
    print("Importing Files...")
    REL_FILE_LOCATION = "bbc"
    all_files=[]
    count=0
    for folders in listdir(REL_FILE_LOCATION):
        if folders != "README.TXT":
            for files in listdir(REL_FILE_LOCATION + "/" + folders):
                count+=1
                current_file = open(REL_FILE_LOCATION + "/" + folders + "/" + files)
                all_files.append([current_file.read(), folders]) #contents of file and folder name(class)
    return all_files

#Creates basic features for the documents.
def feature_eng(documents):
    print("Creating Features...")
    for inst in documents:
        #Tokenize all parts of the document
        tokens_words = word_tokenize(inst[0])
        tokens_sentence = sent_tokenize(inst[0])
        tokens_title = word_tokenize(tokens_sentence[0])

        #Calculate features
        title_length_tokens = len(tokens_title)
        raw_character_count = len(inst[0])
        raw_token_count = len(tokens_words)
        raw_sentence_count =len(tokens_sentence)
        raw_average_token_length = raw_character_count/raw_token_count
        sentence_average_length_tokens = raw_token_count/raw_sentence_count
        sentence_average_length_characters = raw_character_count/raw_sentence_count
        clean = text_cleaner(tokens_words)

        #Insert features
        inst.insert(0, clean) #Clean text
        inst.extend([
            title_length_tokens,
            raw_character_count,
            raw_token_count,
            raw_sentence_count,
            raw_average_token_length,
            sentence_average_length_tokens,
            sentence_average_length_characters,
        ])

#Lemmatizes raw text to return a clean copy.
def text_cleaner(text):
    lemmatizer = WordNetLemmatizer()
    clean_text_array=[]
    for i in text:
        clean_text_array.append(lemmatizer.lemmatize(i.lower())) if i.isalpha() else None
    clean_text=" ".join(clean_text_array)
    return clean_text

#Splits the data between three sets.
def create_splits(documents, SPLITS, SEED):
    print("Creating splits...")
    X_temp = []
    y_temp = []
    #X=clean text + all numerical values (need clean text later)
    for inst in documents:
        X_temp.append([inst[0]] + inst[3:])
        y_temp.append(inst[2])

    #Split twice, Train/Test then Test/Val
    Split1 = 100-SPLITS[0]
    Split2 = 1-(Split1-SPLITS[1])/Split1
    X_train, X_test, y_train, y_test = train_test_split(X_temp, y_temp, test_size=Split1/100, random_state=SEED)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=Split2, random_state=SEED)
    return X_train, X_val, X_test, y_train, y_val, y_test

#Creates vocabulary using the training set
def create_vocab(xtrain, VOCAB_SIZE):
    print("Creating word frequencies...")
    cv = sklearn.feature_extraction.text.CountVectorizer(max_features=VOCAB_SIZE, stop_words="english")
    bag_of_words = []
    for i in xtrain:
        bag_of_words.append(i[0])

    word_count_vector=cv.fit_transform(bag_of_words)
    tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True) 
    tfidf_transformer.fit(word_count_vector)
    return tfidf_transformer, cv

#Creates word frequency vectors for each split
def create_word_freq(all_splits, cv, tf):
    split_vector = []
    for each_split in all_splits:
        document_word_list=[]
        for each_document in each_split:
            document_word_list.append(each_document[0])
        count_vector=cv.transform(document_word_list)
        tf_idf_vector=tf.transform(count_vector)
        tf_idf_vector=tf_idf_vector.toarray().tolist()
        split_vector.append(tf_idf_vector)
    return split_vector

#Combines the previous features with the word frequency vector.
def merge_vector(split_vectors, splits):
    for i, split in enumerate(splits):
        for j, document in enumerate(split):
            document += split_vectors[i][j]
            document.pop(0)
    return splits[0], splits[1], splits[2]

#Test different paramaters and return the best set.
def param_tuning(X_train, X_val, y_train, y_val):
    print("Starting paramater tuning on the development set...")

    #all poss values to be tested
    kernels = ["rbf", "sigmoid", "linear", "poly"]
    cs = [0.001, 0.1, 10, 25, 50, 100, 1000]
    gammas = [1e-2, 1e-3, 1e-4, 1e-5]

    best_params = []
    #all possible combos
    for kernel in kernels:
        for c in cs:
            for gamma in gammas:
                print(kernel, c, gamma)
                #train the model
                svm_clf=sklearn.svm.SVC(kernel=kernel, C=c, gamma=gamma)
                svm_clf.fit(X_train, y_train)
                y_pred = []
                #test on validation set
                for i in range(0, len(X_val)):
                    y_pred.append(svm_clf.predict([X_val[i]]))
                x = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
                #store best params
                if len(best_params) == 0 or best_params[0] < float(x["macro avg"]["f1-score"]):
                    best_params = [float(x["macro avg"]["f1-score"]), kernel, c, gamma]
    print("Best param found: f1-score =", best_params[0], "Kernel=", best_params[1], "C=", best_params[2], "Gamma=", best_params[3])
    return best_params[1], best_params[2], best_params[3]

#Selects the top ranking features upto no_features arg provided, from all features,
def feature_selection(X, y, no_features):
    print("Performing feature selection...")
    fs = SelectKBest(k=no_features)
    fs.fit_transform(X, y)
    column_names = range(0, len(X[0]))
    top_features = sorted(zip(column_names, fs.scores_), key=lambda x: x[1], reverse=True)
    return [item[0] for item in top_features[:no_features]]

#Reduces the features in each split of data to the supplied amount
#uses indexes of the top ranking fearures to remove others.
def reduce_features(splits, safe_features):
    print("Performing feature reductions...")
    x_train = []
    x_val = []
    x_test = []
    #go through each split, doc and feature, keep feature if its in the safe list
    for i, split in enumerate(splits):
        for doc in split:
            temp = []
            for j, feature in enumerate(doc):
                if j in safe_features:
                    temp.append(feature)
            if i == 0:
                x_train.append(temp)
            elif i == 1:
                x_val.append(temp)
            elif i == 2:
                x_test.append(temp)
    return np.asarray(scale(x_train)), np.asarray(scale(x_val)), np.asarray(scale(x_test))

#Test the model against the test set.
def test_model(kernel, c, gamma, X_train, X_test, y_train, y_test, ARGS):
    print("Testing model...")
    svm_clf=sklearn.svm.SVC(kernel=kernel, C=c, gamma=gamma)
    svm_clf.fit(X_train,y_train)

    y_pred = []
    for inst in X_test:
        y_pred.append(svm_clf.predict([inst]))
    print()
    print("Args", ARGS)
    print()
    print(classification_report(y_test, y_pred))

    plot_confusion_matrix(svm_clf, X_test, y_test)
    plt.title("Confusion Matrix")
    plt.xticks(rotation = 90)
    plt.show()  

def main():
    value_check, ARGS, FEATURE_SIZE, VOCAB_SIZE, SPLITS, SEED  = Process_args()

    #if arguments are okay
    if value_check:
        #Import data
        all_files = import_files()
        #create features and split up data
        feature_eng(all_files)
        X_train, X_val, X_test, y_train, y_val, y_test = create_splits(all_files, SPLITS, SEED)
        tf, cv = create_vocab(X_train, VOCAB_SIZE)
        split_vectors = create_word_freq([X_train, X_val, X_test], cv, tf)
        X_train, X_val, X_test = merge_vector(split_vectors, [X_train, X_val, X_test])

        #Feature selection and reduction
        safe_indexs = feature_selection(X_train, y_train, FEATURE_SIZE)
        X_train, X_val, X_test = reduce_features([X_train, X_val, X_test], safe_indexs)

        #Paramater tuning
        kernel, c, gamma = param_tuning(X_train, X_val, y_train, y_val)

        #Test the model and print results.
        test_model(kernel, c, gamma, X_train, X_test, y_train, y_test, ARGS)

if __name__ == "__main__":
    main()