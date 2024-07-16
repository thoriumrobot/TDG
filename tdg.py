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
        self.graph[from_node].append(to_node)

    def to_dict(self):
        return self.graph

def parse_java_file(file_path, tdg):
    with open(file_path, 'r') as file:
        code = file.read()
    tree = javalang.parse.parse(code)

    class TypeVisitor(javalang.tree.Visitor):
        def __init__(self, tdg):
            self.tdg = tdg
            self.current_class = None
            self.current_method = None

        def visit_ClassDeclaration(self, node):
            self.current_class = node.name
            if node.extends:
                parent_class = node.extends.name
                self.tdg.add_dependency(node.name, parent_class)
            else:
                self.tdg.add_dependency(node.name, 'Object')  # Default to Object if no superclass is specified
            self.visit_children(node)

        def visit_MethodDeclaration(self, node):
            method_name = f'{self.current_class}.{node.name}'
            self.current_method = method_name
            for param in node.parameters:
                param_type = self.get_type_name(param.type)
                self.tdg.add_dependency(method_name, param_type)
                self.tdg.add_dependency(param_type, method_name)
            if node.return_type:
                return_type = self.get_type_name(node.return_type)
                self.tdg.add_dependency(return_type, method_name)
                self.tdg.add_dependency(method_name, return_type)
            self.visit_children(node)
            self.current_method = None

        def visit_VariableDeclarator(self, node):
            var_name = node.name
            var_type = self.get_type_name(node.type)
            if self.current_method:
                var_name = f'{self.current_method}.{var_name}'
            else:
                var_name = f'{self.current_class}.{var_name}'
            self.tdg.add_dependency(var_name, var_type)
            self.tdg.add_dependency(var_type, var_name)
            self.visit_children(node)

        def visit_FieldDeclaration(self, node):
            for declarator in node.declarators:
                field_name = f'{self.current_class}.{declarator.name}'
                field_type = self.get_type_name(node.type)
                self.tdg.add_dependency(field_name, field_type)
                self.tdg.add_dependency(field_type, field_name)
                self.visit(declarator)
            self.visit_children(node)

        def visit_LocalVariableDeclaration(self, node):
            for declarator in node.declarators:
                var_name = f'{self.current_method}.{declarator.name}'
                var_type = self.get_type_name(node.type)
                self.tdg.add_dependency(var_name, var_type)
                self.tdg.add_dependency(var_type, var_name)
                self.visit(declarator)
            self.visit_children(node)

        def get_type_name(self, type_node):
            if isinstance(type_node, javalang.tree.ReferenceType):
                return type_node.name
            elif isinstance(type_node, javalang.tree.BasicType):
                return type_node.name
            elif isinstance(type_node, javalang.tree.TypeArgument):
                return self.get_type_name(type_node.type)
            else:
                return 'Unknown'

    visitor = TypeVisitor(tdg)
    for path, node in tree:
        visitor.visit(node)

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

