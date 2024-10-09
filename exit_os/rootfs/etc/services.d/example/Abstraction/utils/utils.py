# Utilities document for common functions, classes and values
import ast
import importlib

def extractClassName(class_code):

    tree = ast.parse(class_code) # Parse the code string into an abstract syntax tree (AST)
    
    # Find the first class definition node (assuming there's only one class in the code)
    class_node = next(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
    
    return class_node.name

def createClass(class_code: str):

    # Define a namespace for the new class
    class_namespace = {}

    # Execute the class content within the namespace
    exec(class_code, globals(), class_namespace)

    # Retrieve the class from the namespace
    class_name = extractClassName(class_code)
    created_class = class_namespace[class_name]

    print(str(created_class))

    return created_class


def isNumber(value):

    if value.isnumeric():
        return True
    else:

        try:
            float(value)
            return True
        except ValueError:
            return False


def toNumber(value):

    if value.__contains__("."):
        return float(value)
    else:
        return int(value)
