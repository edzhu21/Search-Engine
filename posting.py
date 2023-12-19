class Posting:
    def __init__(self, docid):
        self.docid = docid
        self.freq = 1
    
    def __repr__(self):
        return f'[{self.docid}, {self.freq}]'
    
    def add(self):
        self.freq += 1
    
    def get_docid(self):
        return self.docid
