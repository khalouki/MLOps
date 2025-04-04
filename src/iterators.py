
class DatasetIterator:
    def __init__(self, data):
        self.data = data
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.data):
            item = self.data.iloc[self.index].to_dict()
            self.index += 1
            return item
        else:
            raise StopIteration
