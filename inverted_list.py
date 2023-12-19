# creates an inverted list object of posting to do document at a time retrieval
class Inverted_List():
    def __init__(self, li):
        self.inverted_list = li
        self.current_position = 0
    
    def get_current_document(self):
        if (len(self.inverted_list) <= self.current_position):
            return -1
        return self.inverted_list[self.current_position][0]
    
    def move_past_document(self, d):
        while 0 < self.get_current_document() <= d:
            self.current_position += 1
    
    def get_current_freq(self):
        if (len(self.inverted_list) <= self.current_position):
            return 1
        return self.inverted_list[self.current_position][1]

    def __len__(self):
        return len(self.inverted_list)