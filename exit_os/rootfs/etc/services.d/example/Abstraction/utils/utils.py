import ast  # Import the 'ast' module for analyzing Python code.
import importlib  # Module for dynamically loading modules (not used here but may be useful for other functions).

def extractClassName(class_code):
    '''
    Function to extract the name of a class from the given Python code.
    Parses the code into an abstract syntax tree (AST).
    Finds the first class definition node (ClassDef), assuming there's only one class.
    Returns the name of the class.
    '''
    tree = ast.parse(class_code)  # Parse the code string into an abstract syntax tree (AST)
    
    # Find the first class definition node (ClassDef) in the AST. Assumes only one class in the code.
    class_node = next(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
    
    return class_node.name  # Return the class name

def createClass(class_code: str):
    '''
    Function to dynamically create a class from a string of code.
    Defines a namespace where the class will be executed.
    Executes the class code in the provided namespace and retrieves the class.
    Returns the created class.
    '''
    class_namespace = {}  # Define a namespace for the class
    
    # Execute the class code in the global context and within the class namespace
    exec(class_code, globals(), class_namespace)
    
    # Extract the class name using the extractClassName function
    class_name = extractClassName(class_code)
    
    # Retrieve the created class from the namespace
    created_class = class_namespace[class_name]

    print(str(created_class))  # Print the class (for logging or debugging purposes)
    
    return created_class  # Return the created class

def isNumber(value):
    '''
    Function to check if a given value is a number (integer or float).
    Returns True if the value is a number, otherwise False.
    '''
    if value.isnumeric():  # Check if the string is a numeric integer
        return True
    else:
        # If it's not an integer, try to convert it to a float
        try:
            float(value)
            return True
        except ValueError:
            return False

def toNumber(value):
    '''
    Function to convert a numeric string to an integer or a float.
    If the value contains a decimal point, it converts to a float.
    Otherwise, it converts to an integer.
    '''
    if value.__contains__("."):  # If the string contains a decimal point, convert to float
        return float(value)
    else:
        return int(value)  # Otherwise, convert to an integer

