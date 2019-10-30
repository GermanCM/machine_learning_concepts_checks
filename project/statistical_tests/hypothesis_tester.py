"""Class to implement hypothesis tests. 
   Inspired by Allen B. Downey, book ThinkStats
"""
# %%
class HypothesisTest(object):
    def __init__(self, data):
        self.data = data
        self.MakeModel()
        self.actual = self.TestStatistic(data)
    
    def PValue(self, iters=1000):
        """PValue computes the probability of the apparent 
           effect under the null hypothesis.
        
        Keyword Arguments:
            iters {int} -- number of iterations (default: {1000})
        
        Returns:
            [float] -- probability for accepting the null hypothesis (i.e.),
                       not seeing the effect of interest 
        """
        self.test_stats = [self.TestStatistic(self.RunModel())
                           for _ in range(iters)]
        count = sum(1 for x in self.test_stats if x >= self.actual)
        return count / iters
    
    def TestStatistic(self, data):
        raise UnimplementedMethodException()
    
    def MakeModel(self):
        pass
    
    def RunModel(self):
        raise UnimplementedMethodException()
# %%
class DiffMeansPermute(HypothesisTest):
    def TestStatistic(self, data):
        group1, group2 = data
        test_stat = abs(group1.mean() - group2.mean())
        return test_stat
    
    def MakeModel(self):
        import numpy as np

        group1, group2 = self.data
        self.n, self.m = len(group1), len(group2)
        self.pool = np.hstack((group1, group2))

    def RunModel(self):
        import numpy as np
        
        np.random.shuffle(self.pool)
        data = self.pool[:self.n], self.pool[self.n:]
        return data

# %%
