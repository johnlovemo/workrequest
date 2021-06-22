import pandas as pd
import re
import string
import nltk
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.svm import LinearSVC
from sklearn import metrics
#from sklearn.externals import joblib
import time
import pickle


""" pd.show_versions(as_json=False)
print('-------------------- check nltk sklearn version ----------------------------')
nltk_version = nltk.__version__
print(nltk_version)
sklearn_version = sklearn.__version__
print(sklearn_version) """

def wr_model(user_input):
    #df = pd.read_csv('wr_level_csv_ver3.csv') #Consumer_Complaints.csv")
    print('------------- model starting --------')
    print('------------- import test file --------')
    df = pd.read_csv('wr_cy20_final_mapping.csv', encoding='cp1252') 
    mapping = pd.read_csv('wr_category_to_url.csv', encoding='cp1252')
    #print(df.head)

    #col = ['Product', 'Consumer Complaint']
    #df= df[col]
    #df= df[pd.notnull(df['Consumer Complaint'])]
    #df.columns=['Product', 'Consumer_complaint']
    #df['category_id'] = df['Product'].factorize()[0]
    #cat_id_df = df[["Product", "category_id"]].drop_duplicates().sort_values('category_id')
    #cat_to_id = dict(cat_id_df.values)
    #id_to_cat = dict(cat_id_df[['category_id','Product']].values)
    #df.head()
    #tfidf = TfidfVectorizer(sublinear_tf= True, #use a logarithmic form for frequency
    #                    min_df = 5, #minimum numbers of documents a word must be present in to be kept
    #                    norm= 'l2', #ensure all our feature vectors have a euclidian norm of 1
    #                    ngram_range= (1,2), #to indicate that we want to consider both unigrams and bigrams.
    #                    stop_words ='english') #to remove all common pronouns to reduce the number of noisy features

    #features = tfidf.fit_transform(df.Consumer_complaint).toarray()
    #labels = df.category_id
    #features.shape

    #from sklearn.model_selection import train_test_split
    print('----------------- split test file -------------------')
    X_train, X_test, y_train, y_test = train_test_split(df['user_input'], df['category'], random_state= 0)

    #from sklearn.feature_extraction.text import CountVectorizer
    print('------------------- vectorize ---------------------')
    count_vect = CountVectorizer()

    #from sklearn.feature_extraction.text import TfidfTransformer
    X_train_counts = count_vect.fit_transform(X_train)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    #from sklearn.svm import LinearSVC
    clf = LinearSVC().fit(X_train_tfidf, y_train)
    #print(clf.predict(count_vect.transform([user_input])))
    # ------------------------------------------------------------------------------------------------- wr_category = clf.predict(count_vect.transform([user_input]))
    print('----------------- category -------------------')
    #print(type(wr_category))
    #print(wr_category)
    
    #wr_series = pd.DataFrame(wr_category, columns = ['category'])
    #wr_category_df = pd.merge(wr_series, mapping, how='inner', on='category' )
    
    #wr_category_url = wr_category_df['url'].to_string(index=False).strip()
    #print('---------------------- url ------------------------')
    #print(wr_category_url)
    # ----------------------------------------------------------------------------------------------------------------------------------------#    
    
    y_pred = clf.predict(count_vect.transform(X_test))
    print(metrics.classification_report(y_test,y_pred, labels= df.category, target_names=df['category'].unique()))
    
    # --------------------------------- pickling the model to a file ----------------------------------------------------- #
    #pickle.dump(clf, open('nlp_WR20.pickle', 'wb')) 
    pikfile = open('nlp_WR20.pickle', 'rb')
    pk = pickle.load(pikfile)
    wr_category = pk.predict(count_vect.transform([user_input]))
    # ---------------------------------------------------------------------------------- #
    
    #return wr_category_url
    return wr_category
    
    # more ideas
    # based on the input, go straight to the correct form
    # as an user logs in, suggest a task, PTA
    # prepopulate building, floor etc.

                                                                                                   