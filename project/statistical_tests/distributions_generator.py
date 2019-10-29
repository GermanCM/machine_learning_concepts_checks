class Distributions_generator():
    
    def __init__(self, dataframe=None):
        self.dataframe = dataframe

    def generate_random_normal_distribution(self, mean_value, 
                                            std_value, size):
        # generate related variables
        from numpy import mean, std
        from numpy.random import randn, seed
        # seed random number generator
        seed(1)
        # generate data
        data_normal_1 = std_value * randn(size) + mean_value

        return data_normal_1 

    def generate_random_distribution(self, std_value, size):
        # generate related variables
        from numpy import mean, std
        from numpy.random import rand, seed
        # seed random number generator
        seed(1)
        # generate data
        data_random = std_value * rand(size)

        return data_random 
