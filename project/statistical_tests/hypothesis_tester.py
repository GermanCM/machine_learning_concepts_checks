"""Class to implement hypothesis tests. 
   Inspired by Allen B. Downey, book ThinkStats
"""
# %%
class HypothesisTest(object):
    """Tests a *null hypothesis* by computing the probability of 
       seeing an effect under such hypothesis: E.g:
       - hypothesis: the effect is only apparent (not statistically significant) 
                     between the two samples (i.e. come from the same distribution)
       - effect: there is enough difference between the samples values to reject
                 that null hypothesis
       - PValue: probability of seeing that effect in case the hypothesis
                 is true; a high pvalue means accepting the null hypothesis,
                 whereas a low pvalue means rejecting the null hypothesis (since 
                 we cannot trust a tiny probability for being just a stistical fluke) 
                 and hence accepting the apparent effect
    
    Raises:
        UnimplementedMethodException: TestStatistic method must be define in a child class
        UnimplementedMethodException: RunModel method must be define in a child class
    
    Returns:
        [type] -- [description]
    """
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
        """Statistic value used to test the considered hypothesis.
           This is the heart of the test, being the value checked
           in each iteration and compared to the reference value
           to compute the probability.
        
        Arguments:
            HypothesisTest {parent class object} -- the inherited class
            data {tuple} -- the pair of sequences on which to apply the test 
        
        Returns:
            [float] -- the t-statistic value
        """
        group1, group2 = data
        test_stat = abs(group1.mean() - group2.mean())
        return test_stat
    
    def MakeModel(self):
        """Joins the two sequences of data into a single group
        """
        import numpy as np

        group1, group2 = self.data
        self.n, self.m = len(group1), len(group2)
        self.pool = np.hstack((group1, group2))

    def RunModel(self):
        """Permutes the data in the sequences
        
        Returns:
            [array] -- a single sequence of permuted data
        """
        import numpy as np
        
        np.random.shuffle(self.pool)
        data = self.pool[:self.n], self.pool[self.n:]
        return data

# %%
