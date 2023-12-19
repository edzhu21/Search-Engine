# IR23F Assignment 3

## Step-by-Step Guide
    1. Run indexer.py (this creates all the necessary files needed at query / runtime), This will prompt you to put in the file path of the data (developer/DEV)
    2. Run query.py (this should launch a GUI)
    3. In the box, enter your query
    4. Click search and the top 5 results of your query should show up in the box underneath
    5. Clicking clear will clear the query box aand the result box

## Milestone 3
Program advances and is able to handle more complex queries due to the addition of relevance score functions. The search engine is able to run queries with a GUI in under 300ms with optimizations.

## Milestone 2
Program additionally takes in a query and does boolean AND search on the index. It then returns the top 5 URLs.

## Milestone 1
Program takes in a data source (developer.zip) and creates an inverted index mapping tokens to Postings (doc id, word frequency). It writes the index to disk from memory in case there are too much data to store in memory.

## Team Members

- Edgar Zhu [@edzhu21](https://www.github.com/edzhu21)
- Waylon Zhu [@Wayloncode](https://www.github.com/Wayloncode)
