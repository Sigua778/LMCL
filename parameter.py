
class para:
    def __init__(self, lr, rec, drop, batchSize, epoch, embed_size, dev_ratio,test_ratio, n_layers,
                 ssl_reg, ssl_ratio, ssl_temp, num_negatives, stddev, stop_cnt, heads, BCE_L):
        self.lr = lr
        self.rec = rec
        self.drop = drop
        self.batchSize = batchSize
        self.epoch = epoch
        self.embed_size = embed_size
        self.dev_ratio = dev_ratio
        self.test_ratio = test_ratio
        self.n_layers = n_layers
        self.ssl_reg = ssl_reg
        self.ssl_ratio = ssl_ratio
        self.ssl_temp = ssl_temp
        self.ssl_reg = ssl_reg
        self.num_negatives = num_negatives
        self.stddev = stddev
        self.stop_cnt = stop_cnt
        self.heads = heads
        self.BCE_L = BCE_L
