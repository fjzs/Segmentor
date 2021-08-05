def myfunction(param1:int, param2:str = ""):
    """
    What does this function do?
    
    Arguments:
    - param1 (int): very specific definition of this parameter
    - param2 (str): very specific definition of this parameter
    
    Returns:
    - output (str): very specific definition of this variable
    """
    
    output = str(param1) + param2
    print(output)


myfunction(3,"test")