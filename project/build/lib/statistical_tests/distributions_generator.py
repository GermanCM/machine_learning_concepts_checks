class Distributions_generator():
    
    def __init__(self, dataframe=None):
        self.dataframe = dataframe

    def generate_random_normal_distribution(self, mean_value, 
                                            std_value, size):
        """Generates a random normal distribution, provided its
           mean and standard deviation

        Arguments:
            mean_value {float} -- the distribution mean value
            std_value {float} -- the distribution standard deviation value
            size {int} -- the distribution number of elements 
        
        Returns:
            array -- the generated distribution
        """
        try:
            # generate related variables
            from numpy import mean, std
            from numpy.random import randn, seed
            # seed random number generator
            #seed(1)
            # generate data
            data_normal_1 = std_value * randn(size) + mean_value

            return data_normal_1 
        except Exception as exc:
            raise exc 

    def generate_random_distribution(self, scaler_value, size):
        """Generates a random distribution, provided its
           mean and standard deviation
        
        Arguments:
            scaler_value {float} -- value to multiply the 
                                    distribtuion elements
            size {int} -- distribution number of elements
        
        Returns:
            array -- the generated distribution
        """
        try:
            from numpy import mean, std
            from numpy.random import rand, seed
            # seed random number generator
            #seed(1)
            # generate data
            data_random = scaler_value * rand(size)

            return data_random 
        except Exception as exc:
            raise exc

    def generate_beta_distribution(self, a, b, size, seed_value=None):
        """Generates a random beta distribution, 
           given the parameters a and b
        
        Arguments:
            a {float or array_like of floats} -- alpha > 0
            b {float or array_like of floats} -- beta > 0
            size {int} -- distribution length 
            
        Keyword Arguments:
            seed_value {int} -- seed random number generator (default: {None})
        
        Raises:
            exc: general exception
        
        Returns:
            array -- generated distribution
        """
        try:
            import numpy as np
            from numpy.random import beta, seed
            # seed random number generator
            if seed_value is not None:
                seed(seed_value)
            # generate data        
            data_random_beta = np.random.beta(a, b)

            return data_random_beta

        except Exception as exc:
            raise exc
