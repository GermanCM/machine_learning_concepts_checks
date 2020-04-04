# %%
"""
This class's goal is to implement methods for efficient operations when dealing with 
iterations.    
"""
class Iterations_operator():
    def __init__(self, _data_container):
        self.data_container = _data_container

    def operation_function(self, parameter_list):
        raise NotImplementedError

    def apply_operation_to_all_elements(self, operation_to_apply):
        """Applies an operation to all elements of a dataframe or series
        
        Arguments:
            operation_function {function} -- this must be a function requiring
                                             an input value 
        
        Raises:
            exc: [description]
        
        Returns:
            [type] -- [description]
        """
        try:
            import pandas as pd 
            if type(self.data_container)==pd.core.frame.DataFrame:
                return self.data_container.applymap(operation_function)
            elif type(self.data_container)==pd.core.frame.Series:
                return self.data_container.apply(operation_function)

        except Exception as exc:
            raise exc

    