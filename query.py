from indexer import get_term_position, load_url_index, read_specific_line
from priority_queue import PQ
from inverted_list import Inverted_List
from nltk.stem.snowball import SnowballStemmer
import re
import math
from collections import defaultdict
from statistics import stdev, mean
import time
from tkinter import *
# for all terms w in Q, find postings in index with w
# need index -> url (url_index)

def g(Q, df, n):
    base = 10
    # function to compute the score for the query
    freq = defaultdict(int)
    for t in Q:
        freq[t] += 1
    
    tf_wt = []
    for tf_raw in freq.values():
        tf_wt.append((1 + math.log(tf_raw, base)) if tf_raw > 0 else 0)
    
    idf = [math.log(n / q_df, base) for q_df in df]
    wt = []
    for i in range(len(tf_wt)):
        wt.append(tf_wt[i] * idf[i])

    return wt

def f(li):
    base = 10
    df = 1
    # function to compute the score the inverted list
    # using lnc
    tf_raw = li.get_current_freq()
    tf_wt = (1 + math.log(tf_raw, base)) if tf_raw > 0 else 0
    wt = tf_wt * df
    wt = tf_raw
    return wt

def normalize(vector):
    # normaliuzes the weight vector for cosine similarity
    norm_value = math.sqrt(sum(num ** 2 for num in vector))
    
    return [num / norm_value for num in vector]

def dot(vector1, vector2):
    # assuming vector1 and vector2 are normalized (they have same length),
    # compute dot product of both vectors

    result = 0
    for i in range(len(vector1)):
        result += vector1[i] * vector2[i]
    
    return result

def document_retrieval(Q, I, location_index, tier, f = f, g = g, k = 5):
    # computes and retrieves top k urls for query Q
    print('Processing your query...')
    # container to eliminate low-idf indexes
    index_idf = dict()
    # container to map word -> postings of query (temporary / unfiltered)
    temp_L = dict()
    L = []
    R = PQ(k)
    q_df = []
    for word in Q:
         
        with open(f'tier{tier}_index.txt', 'r', encoding = 'utf-8') as index:
            try:
                location = location_index[word]
            except KeyError:
                return dict()

            d = read_specific_line(f'tier{tier}_index.txt', location)

            document_list = d[word]
            

            index_idf[word] = math.log(len(I) / len(document_list), 10)

            temp_L[word] = document_list
            q_df.append(len(document_list))
        
        
    z_score = 0
    if len(Q) > 1:
        mean_idf = mean(list(index_idf.values()))
        std_dev_idf = stdev(list(index_idf.values()))

    for word, idf in index_idf.items():
        if len(Q) > 1:
            z_score = (idf - mean_idf) / std_dev_idf
        # prune indexes of query that contribute little to score because low idf
        # threshold set to 0.8 std deviations to the left of mean idf
        if z_score >= -0.8:
            L.append(Inverted_List(temp_L[word]))
        else:
            print(word)

    q_wt = g(Q, q_df, len(I))
    norm_query_wt = normalize(q_wt)

    # sort the inverted lists by tf
    L.sort(key=len)

    num_terms = len(Q)
    
    for doc_id, url in I.items():
        score = 0
        doc_wt = []

        for li in L:
            if li.get_current_document() == doc_id:
                #score += g(Q)*f(li)
                doc_wt.append(f(li))
            
            li.move_past_document(doc_id)

        if  len(doc_wt) == 0 or num_terms / len(doc_wt) < 0.5:
            continue

        norm_doc_wt = normalize(doc_wt)
        score = dot(norm_doc_wt, norm_query_wt)

        if score > 0:
            R.push(doc_id, score)
    print(len(R.results()))
    return R.results()[:k]


if __name__ == '__main__':
    url_index = load_url_index()
    t1_location_index = get_term_position('tier1_index.txt')
    t2_location_index = get_term_position('tier2_index.txt')
    while True:
        snow_stemmer = SnowballStemmer(language='english')

        root = Tk()
        root.title("Search Engine")
        root.geometry('1000x500')
        lbl = Label(root, text="Please input your query:")
        lbl.pack()

        #query = input('Please input your query: ')
        txt = Entry(root, width=50)
        txt.pack()#(column = 1, row=0)

        def clicked():
            '''calls backend function of search engine and gets top 5 results and puts 
            it in result box of GUI
            '''
            query = txt.get()
            # start the timer
            start_time = time.time()
            q = [snow_stemmer.stem(word.lower()) for word in re.split('[^a-z0-9A-Z]+', query)]
            print('Top 5 results are: ')

            urls = [url_index[d_id] for _, d_id in document_retrieval(q, url_index, t1_location_index, 1)]
            if len(urls) < 5:
                urls += [url_index[d_id] for _, d_id in document_retrieval(q, url_index, t2_location_index, 2)]
            
            for link in urls:
                T.insert(END, link+"\n")

            # Stop the timer
            end_time = time.time()
            print(urls)
            elapsed_time = end_time - start_time
            results = f"Found in {round(elapsed_time * 1000)} ms."
            print(results)
            T.insert(END, results + "\n")

        def clear():
            # clears text entry and result box in GUI
            txt.delete(0, END)
            T.delete("1.0","end")
    
        T = Text(root, height=20, width=100)

        btn = Button(root, text="Search", fg="blue", command=clicked)
        btn.pack()
        btn2 = Button(root, text="Clear", fg="blue", command=clear)
        btn2.pack()
        T.pack()
        root.mainloop()
        break