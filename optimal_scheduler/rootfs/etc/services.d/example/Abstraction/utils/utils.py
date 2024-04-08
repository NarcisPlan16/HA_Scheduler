# Utilities document for common functions, classes and values

import importlib


def createClass(asset_class, asset_config):

    # Split the class string into class name and content
    class_lines = asset_class.strip().split('\n')
    print(class_lines)
    class_name = class_lines[0].split(' ')[1]
    class_content = '\n'.join(class_lines[1:])

    # Define a namespace for the new class
    class_namespace = {}

    # Execute the class content within the namespace
    exec(class_content, globals(), class_namespace)

    # Retrieve the class from the namespace
    created_class = class_namespace[class_name]

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
