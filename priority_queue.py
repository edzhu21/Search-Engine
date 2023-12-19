import heapq
# Priority QUeue to contain the (docids, scores) of postings from query
class PQ():
    def __init__(self, max_size):
        self.pq = []
        self.max_size = max_size
    
    def push(self, document, priority):
        heapq.heappush(self.pq, (priority, document))
    
    def pop(self):
        print(heapq.heappop(self.pq))
    
    def is_full(self):
        return len(self.pq) == self.max_size

    def results(self):
        #print(self.pq)
        #return sorted(self.pq, key = lambda x: x[0])[:self.max_size]
        return heapq.nsmallest(len(self.pq), self.pq)

    def __repr__(self):
        return f'Priority Queue: {self.pq}'