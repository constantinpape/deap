class MixDataset(object):

    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.split = len(self.dataset1)
        # self.step = step

    def __len__(self):
        return len(self.dataset1) + len(self.dataset2)
    
    def __getitem__(self, idx):
        if idx < self.split:
            return self.dataset1[idx]
        else:
            return self.dataset2[idx - self.split]


class Mix3Dataset(object):

    def __init__(self, dataset1, dataset2, dataset3):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.dataset3 = dataset3
        self.split1 = len(self.dataset1)
        self.split2 = self.split1 + len(self.dataset2)
        # self.step = step

    def __len__(self):
        return len(self.dataset1) + len(self.dataset2) + len(self.dataset3)
    
    def __getitem__(self, idx):
        if idx < self.split1:
            return self.dataset1[idx]
        elif idx < self.split2:
            return self.dataset2[idx - self.split1]
        else:
            return self.dataset3[idx - self.split2]


class BaseDataset(object):

    def __init__(self):
        super().__init__()  

    def get_sample(seq_idx, indices):
        raise NotImplementedError

    def n_sequences(self):
        raise NotImplementedError

    def sequence_length(self, seq_idx):
        raise NotImplementedError

    def __len__(self):
        return self.n_sequences() * 100


class CachedDataset(object):
    def __init__(self, dataset, keys=None, types=None):
        self.dataset = dataset
        self.keys = keys
        self.types = types
        self.cached_data = dict()

    def __getitem__(self, index):
        if index not in self.cached_data:
            s = self.dataset[index]
            if self.keys is not None:
                s = {k: v for k,v in s.items() if k in self.keys}

            if self.types is not None:
                s = {k: v.to(self.types[k]) if k in self.types else v for k,v in s.items()}

            self.cached_data[index] = s
            return s
        else:
            return self.cached_data[index]

    def __len__(self):
        return len(self.dataset)
    