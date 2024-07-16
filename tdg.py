import os
import javalang
import json
import argparse

class TypeDependencyGraph:
    def __init__(self):
        self.graph = {}

    def add_dependency(self, from_node, to_node):
        if from_node not in self.graph:
            self.graph[from_node] = []
        if to_node not in self.graph[from_node]:
            self.graph[from_node].append(to_node)

    def to_dict(self):
        return self.graph

def parse_java_file(file_path, tdg):
    with open(file_path, 'r') as file:
        code = file.read()
    tree = javalang.parse.parse(code)

    def get_type_name(type_node):
        if isinstance(type_node, javalang.tree.ReferenceType):
            return type_node.name
        elif isinstance(type_node, javalang.tree.BasicType):
            return type_node.name
        elif isinstance(type_node, javalang.tree.TypeArgument):
            return get_type_name(type_node.type)
        else:
            return 'Unknown'

    current_class = None
    current_method = None

    for path, node in tree:
        if isinstance(node, javalang.tree.ClassDeclaration):
            current_class = node.name
            if node.extends:
                parent_class = node.extends.name
                tdg.add_dependency(node.name, parent_class)
            else:
                tdg.add_dependency(node.name, 'Object')
        elif isinstance(node, javalang.tree.MethodDeclaration):
            method_name = f'{current_class}.{node.name}'
            current_method = method_name
            for param in node.parameters:
                param_type = get_type_name(param.type)
                tdg.add_dependency(method_name, param_type)
            if node.return_type:
                return_type = get_type_name(node.return_type)
                tdg.add_dependency(method_name, return_type)
            tdg.add_dependency(current_class, method_name)
        elif isinstance(node, javalang.tree.MethodInvocation):
            invoked_method = f'{node.qualifier}.{node.member}' if node.qualifier else f'{current_class}.{node.member}'
            tdg.add_dependency(current_method, invoked_method)
        elif isinstance(node, javalang.tree.VariableDeclarator):
            var_name = node.name
            if current_method:
                var_name = f'{current_method}.{var_name}'
            else:
                var_name = f'{current_class}.{var_name}'
            var_type = 'Unknown'
            parent_node = path[-1]
            if isinstance(parent_node, javalang.tree.FieldDeclaration) or isinstance(parent_node, javalang.tree.LocalVariableDeclaration):
                var_type = get_type_name(parent_node.type)
            tdg.add_dependency(var_name, var_type)
        elif isinstance(node, javalang.tree.FieldDeclaration):
            for declarator in node.declarators:
                field_name = f'{current_class}.{declarator.name}'
                field_type = get_type_name(node.type)
                tdg.add_dependency(field_name, field_type)

def create_tdg_from_directory(directory):
    tdg = TypeDependencyGraph()

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.java'):
                file_path = os.path.join(root, file)
                parse_java_file(file_path, tdg)

    return tdg

def save_tdg_to_json(tdg, output_file):
    with open(output_file, 'w') as file:
        json.dump(tdg.to_dict(), file, indent=4)

def main():
    parser = argparse.ArgumentParser(description='Create a Type Dependency Graph from a Java project directory.')
    parser.add_argument('directory', help='Path to the Java project directory')
    parser.add_argument('output', help='Path to the output JSON file')
    args = parser.parse_args()

    tdg = create_tdg_from_directory(args.directory)
    save_tdg_to_json(tdg, args.output)

if __name__ == '__main__':
    main()

