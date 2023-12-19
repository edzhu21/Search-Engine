from bs4 import BeautifulSoup
import os
import json
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from posting import Posting
import psutil
from collections import defaultdict, deque
import math
from urllib.parse import urljoin
import pickle

# global variable to store the outgoing links of dataset
outgoing_links = defaultdict(set)
# global variable to keep track of recent 100 document fingerprints (urls)
queue = deque(maxlen = 100)
def create_n_grams(tokens, n = 3):
    # function to create n-grams of the tokens
    n_grams = []
    for i in range(len(tokens) - n):
        n_grams.append(tokens[i:i+n])

    return n_grams

def similarity_detection(n_grams, k = 4, threshold = 0.8):
    # given n-grams, compute hash values, select certain hash-values, then check if document fingerprint is similar to previous 100 URLs
    # n-grams: sequences of n tokens of current document
    # k: mod value; subject to change
    hash_values = []
    for gram in n_grams:
        h = hash(tuple(gram))
        if h % k == 0:
            hash_values.append(h)

    global queue
    # if there is enough overlap between current document and previous 100 document fingerprints, document is similar -> can ignore
    for fingerprint in queue:
        if overlap(hash_values, fingerprint) > threshold:
            return True
    # no overlap
    queue.append(hash_values)
    return False

def overlap(current, recent):
    # current: current document fingerprint
    # recent: recent document fingerprints
    # returns the overlap of the document fingerprints
    overlap = len([1 for i in current if i in recent])
    total = len([1 for i in current if i not in recent]) + len([1 for i in recent if i not in current]) + overlap
    if total == 0:
        return 0
    return overlap / total
def parse(content):
    # parses document assuming its in html using BeautfilSoup
    # tokenizes the text after parsing
    soup = BeautifulSoup(content, 'html.parser')
    text = soup.get_text()
    snow_stemmer = SnowballStemmer(language='english')
    tokens = [snow_stemmer.stem(t.lower()) for t in word_tokenize(text) if t.isalnum()]
    return tokens

def posting_exists(postings_list, posting):
    # checks if the posting already exists
    for p in postings_list:
        if p.docid == posting.docid:
            return p
    return None

def get_batch(D, batch_size = 15000):
    # gets a batch of size batch_size from the total documents
    b = []
    for root, dirs, files in os.walk(D):
        for name in files:
            file_path = os.path.join(root, name)
            with open(file_path, 'r') as data:
                d = json.load(data)
            
            b.append(d)
            print(f'Total pages: {len(b)}')
            if len(b) >= batch_size:
                
                yield b
                b = []
    if b:
        yield b
def build_graph(d):
    '''
    Constructs the graph of the webpages and its outgoing links
    '''
    soup = BeautifulSoup(d['content'].lower(), "html.parser")
    links = soup.find_all('a')

    for link in links:
        if link.has_attr('href'):
            abs = urljoin(d['url'], link['href'])
            outgoing_links[d['url']].add(abs)
def pagerank(graph, num_iterations=100, damping_factor=0.85):
    # number of nodes in graph
    n = len(graph)
    p = {node: 1/n for node in graph}

    for i in range(num_iterations):
        print(i)
        new_p = defaultdict(float)
        for node in graph:
            for in_node in graph:
                if node in graph[in_node]:  # Check if there is a link from in_node to node
                    outgoing_edges = len(graph[in_node])
                    L_ij = 1 if in_node in graph[node] else 0
                    new_p[node] += (1 - damping_factor) + damping_factor * (L_ij / outgoing_edges * p[in_node])
        p = new_p
    print('converged PR')
    return p
def hits(graph, tolerance=0.001, max_iterations=100):
    '''
    Computes the hub and authority scores for each node in the graph
    '''
    n = len(graph)
    hubs = {node: 1/n for node in graph}
    authorities = {node: 1/n for node in graph}

    for _ in range(max_iterations):
        new_hubs = defaultdict(float)
        new_authorities = defaultdict(float)

        for node in graph:
            for out_node in graph[node]:
                new_hubs[out_node] += authorities[node] / len(graph[out_node])
                new_authorities[node] += hubs[out_node] / len(graph[out_node])

        hubs_delta = sum(abs(hubs[node] - new_hubs[node]) for node in graph)
        authorities_delta = sum(abs(authorities[node] - new_authorities[node]) for node in graph)

        if hubs_delta < tolerance and authorities_delta < tolerance:
            break

        hubs = new_hubs
        authorities = new_authorities

    return hubs, authorities
def build_index(D):
    # creates an inverted index mapping tokens to Posting(docid, word_freq)
    # stores index from memory into disk after a cretain amount of batches
    global url_index
    index = dict()
    n = 0 # doc id
    batch = []
    DEBUG = 0
    counter = 0
    url_index = open('url_index.txt', 'w', encoding = 'utf-8')

    for batch in get_batch(D):
        # for each document in batch
        for d in batch:
            n += 1
            url_index.write(f"{n}: {d['url']}\n")

            tokens = parse(d['content'])
            n_grams = create_n_grams(tokens)
            if similarity_detection(n_grams):
                continue
            # for each term in tokens
            for e in tokens:
                # create posting
                posting = Posting(n)
        
                if e not in index:
                    index[e] = [posting]
                else:
                    if index[e][-1].get_docid() == n:
                        index[e][-1].add()
                    else:
                        index[e].append(posting)

            build_graph(d)
        
        if True:
            create_partial_index(index, counter)
            index = dict()
            counter += 1
            #current_usage = psutil.virtual_memory().percent
            
        # if DEBUG > 3:
        #     break
        # else:
        #     DEBUG+=1
    
    url_index.close()
    # creates partial index on disk and empties the index

total_doc = 0
unique_words = 0

def create_partial_index(index, partial_index_count):
    # sort index alphabetically
    index = dict(sorted(index.items()))
    # create new partial index text files
    with open("partial_index" + str(partial_index_count) + ".txt", "w", encoding = 'utf-8') as file:
        # write contents of index dict onto text file
        for key, values in index.items():
            key_string = '{"' + key
            file.write(key_string)
            file.write('": ')
            file.write(str(values))
            file.write("}\n")
    # empty index dict
    index.clear() 

# merges and sorts the partial indexes
def merge_and_sort_indexes():
    # 4 partial indexes since we do 15000 docs in one batch
    num_partial_indexes = 4
    counter = 0
    # used to store the content in each line
    line = [None] * num_partial_indexes
    # used to store the tokens in each line
    token_dict = [None] * num_partial_indexes

    # open the files simultaneously and read from all of them
    file_list = [open("partial_index{}.txt".format(x), "r", encoding='utf-8') for x in range(num_partial_indexes)]

    # read one line from each partial index and add it to the token_dict
    while counter < num_partial_indexes:
        line[counter] = file_list[counter].readline()
        token_dict[counter] = json.loads(line[counter])
        counter += 1

    # flag to control merging loop 
    loop = True

    # list that tracks indexes with remaining content
    valid_i = [i for i in range(num_partial_indexes)]

    while loop:
        # get the minimum token among the current tokens in each partial idnex
        token = min(list(token_dict[x].keys())[0] for x in valid_i)

        # new dictionary to store merged entries with the same token
        new_dict = defaultdict(list)

        # iterate over the partial indexes and merge entries with the same token
        for index in valid_i:
            if list(token_dict[int(index)].keys())[0] == token:
                for element in token_dict[index][token]:
                    new_dict[token].append(element)
                # read the next line for that partial index
                line[index] = file_list[index].readline()
                if not line[index]:
                    valid_i.remove(index)
                    file_list[index].close()
                else:
                    # update token with the next line's content
                    token_dict[index] = json.loads(line[index])

        # write the merged dictionary onto disk
        with open("combined_index.txt", "a", encoding='utf-8') as f:
            f.write(str(dict(new_dict)))
            f.write("\n")

        # check if all valid indexes have been processed
        if not valid_i:
            loop = False


# store each term and its line location in memory
def get_term_position(text_file):
    # dict used to store term : line location
    term_positions = {}
    with open(text_file, "rb") as file:
        current_position = 0
        for line in file:
            # get the term from each line
            temp_line = line.decode('utf-8').strip().split(': ')
            term = temp_line[0][2:len(temp_line[0])-1]
            # add the term and its line location to the term_positions dict 
            term_positions[term] = current_position
            # increment the current position
            current_position += len(line)

    return term_positions

            # use this if we want to write the location index into file
            # with open("t1.txt", "a", encoding='utf-8') as f:
            #     f.write(str(term_positions))
            #     f.write('\n')
            #     term_positions.clear()
def load_url_index():
    '''
    loads the url index (maps docid to url) from txt to dict
    '''
    result = dict()
    with open('url_index.txt', 'r', encoding = 'utf-8') as url_index:
        lines = url_index.read().split('\n')
        for line in lines:
            if line:
                key, value = line.split(': ')
                key = int(key)
                result[key] = value

    return result

def read_specific_line(text_file, seek_num):
    # open the full index file and read it as bites
    with open(text_file, "r") as file:
        # jump to location of the term we want
        file.seek(seek_num, 0)
        line = file.readline().strip()
        line = line.replace("'", "\"")
        posting_dict = json.loads(line)

    return posting_dict

def compute(tf, idf, url_index):
    # computes the ifidf score and returns it
    return (1 + math.log(tf, 10)) * math.log((len(url_index) / idf), 10)

def change_index():
    # change index postings so instead of storing doc id and term freq,
    # it now stores doc id and the tfidf score
    new = dict()
    url_index = load_url_index()
    with open('combined_index.txt', 'r', encoding='utf-8') as index:
        for line in index:
            line = eval(line)
            for key, value in line.items():
                new[key] = []
                for doc_id, tf in value:
                    tfidf = compute(tf, len(line), url_index)
                    new[key].append([doc_id, tfidf])
    with open('combined_index.txt', 'w', encoding='utf-8') as index:
        for key, values in new.items():
                key_string = '{"' + key
                index.write(key_string)
                index.write('": ')
                index.write(str(values))
                index.write("}\n")

# split the merged index into tier1 and tier2 indexes       
def split_index():
    # read from combined_index.txt
    with open("combined_index.txt", "r", encoding='utf-8') as file:
        line = file.readline()

        while line:
            # get the dictionary for that term
            d = eval(line)
            t1_dict = defaultdict(list)
            t2_dict = defaultdict(list)
            for key, values in d.items():
                for l in values:
                    # if the tf-idf is bigger than 6, add to tier 1
                    if l[1] > 6:
                        t1_dict[key] += [l]
                    else:
                    # add to tier 2 index if tf-idf is less than 6
                        t2_dict[key] += [l]

                # write the dict into tier1_index.txt
                with open("tier1_index.txt", "a", encoding='utf-8') as t1:
                    if len(t1_dict) != 0:
                        t1.write(str(dict(t1_dict)))
                        t1.write("\n")
                
                # write the dict into tier2_index.txt
                with open("tier2_index.txt", "a", encoding='utf-8') as t2:
                    if len(t2_dict) != 0:
                        t2.write(str(dict(t2_dict)))
                        t2.write("\n")
                # read the next line
                line = file.readline()

if __name__ == '__main__':
    # path = 'C:/Users/edzhu/Desktop/School/CS 121/IR23F-A3-G61/developer/DEV'
    path = input('Path of data:')
    build_index(path)
    merge_and_sort_indexes()
    # changes the merged index so that isntead of term -> docid, freq, it is now
    # term -> docid, tfidf
    change_index()
    split_index()
    #print(len(outgoing_links))
    # with open('graph.pkl', 'wb') as file:
    #     pickle.dump(outgoing_links, file)
    # test = pagerank(outgoing_links)
    # with open('pagerank_results.pkl', 'wb') as file:
    #     pickle.dump(test, file)
    