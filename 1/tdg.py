import os
import sys
import json
import javalang
from collections import defaultdict

class TypeDependencyGraph:
    def __init__(self):
        self.graph = defaultdict(lambda: {"depends_on": [], "type": None})

    def add_node(self, node, node_type):
        if node not in self.graph:
            self.graph[node] = {"depends_on": [], "type": node_type}

    def add_edge(self, from_node, to_node):
        if to_node not in self.graph[from_node]["depends_on"]:
            self.graph[from_node]["depends_on"].append(to_node)

    def to_dict(self):
        return self.graph

    def save_to_json(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

def parse_java_file(file_path):
    try:
        with open(file_path, 'r') as file:
            return javalang.parse.parse(file.read())
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None

def process_java_file(file_path, tdg):
    tree = parse_java_file(file_path)
    if tree is None:
        return

    for path, node in tree:
        if isinstance(node, javalang.tree.ClassDeclaration):
            class_name = node.name
            tdg.add_node(class_name, "Class")
            process_class_body(node, class_name, tdg)

def get_type_name(type_node):
    if type_node is None:
        return None
    if isinstance(type_node, javalang.tree.BasicType):
        return type_node.name
    if isinstance(type_node, javalang.tree.ReferenceType):
        return '.'.join(type_node.name) if isinstance(type_node.name, list) else type_node.name
    if isinstance(type_node, javalang.tree.ClassReference):
        return type_node.name
    if isinstance(type_node, javalang.tree.TypeArgument):
        return get_type_name(type_node.type)
    return None

def process_class_body(node, class_name, tdg):
    if node.fields:
        for field in node.fields:
            field_type = get_type_name(field.type)
            for decl in field.declarators:
                field_name = f"{class_name}.{decl.name}"
                tdg.add_node(field_name, field_type)
                tdg.add_edge(class_name, field_name)
    if node.methods:
        for method in node.methods:
            method_name = f"{class_name}.{method.name}"
            tdg.add_node(method_name, "Method")
            if method.return_type:
                return_type = get_type_name(method.return_type)
                tdg.add_node(f"{method_name}.return", return_type)
                tdg.add_edge(method_name, f"{method_name}.return")
            if method.parameters:
                for param in method.parameters:
                    param_name = f"{method_name}.{param.name}"
                    param_type = get_type_name(param.type)
                    tdg.add_node(param_name, param_type)
                    tdg.add_edge(method_name, param_name)
            if method.body:
                process_method_body(method.body, method_name, tdg)
    if node.constructors:
        for constructor in node.constructors:
            constructor_name = f"{class_name}.{constructor.name}"
            tdg.add_node(constructor_name, "Constructor")
            if constructor.parameters:
                for param in constructor.parameters:
                    param_name = f"{constructor_name}.{param.name}"
                    param_type = get_type_name(param.type)
                    tdg.add_node(param_name, param_type)
                    tdg.add_edge(constructor_name, param_name)
            if constructor.body:
                process_method_body(constructor.body, constructor_name, tdg)

def process_method_body(body, method_name, tdg):
    for path, node in body:
        if isinstance(node, javalang.tree.LocalVariableDeclaration):
            for decl in node.declarators:
                var_name = f"{method_name}.{decl.name}"
                var_type = get_type_name(node.type)
                tdg.add_node(var_name, var_type)
                tdg.add_edge(method_name, var_name)
        elif isinstance(node, javalang.tree.MethodInvocation):
            invoc_name = f"{method_name}.{node.member}"
            tdg.add_node(invoc_name, "MethodInvocation")
            tdg.add_edge(method_name, invoc_name)
            if node.arguments:
                for arg in node.arguments:
                    process_method_argument(arg, invoc_name, tdg)
        elif isinstance(node, javalang.tree.Assignment):
            if isinstance(node.expressionl, javalang.tree.MemberReference):
                assigned_var = f"{method_name}.{node.expressionl.member}"
                assigned_type = get_type_name(node.type)
                tdg.add_node(assigned_var, assigned_type)
                tdg.add_edge(method_name, assigned_var)
            elif isinstance(node.expressionl, javalang.tree.VariableDeclarator):
                assigned_var = f"{method_name}.{node.expressionl.name}"
                assigned_type = get_type_name(node.type)
                tdg.add_node(assigned_var, assigned_type)
                tdg.add_edge(method_name, assigned_var)
        elif isinstance(node, javalang.tree.ClassCreator):
            creator_type = get_type_name(node.type)
            creator_name = f"{method_name}.{creator_type}"
            tdg.add_node(creator_name, "ClassCreator")
            tdg.add_edge(method_name, creator_name)
        elif isinstance(node, javalang.tree.MemberReference):
            member_name = f"{method_name}.{node.member}"
            tdg.add_node(member_name, "MemberReference")
            tdg.add_edge(method_name, member_name)

def process_method_argument(arg, invoc_name, tdg):
    if isinstance(arg, javalang.tree.Literal):
        arg_name = f"{invoc_name}.{arg.value}"
        tdg.add_node(arg_name, "Literal")
        tdg.add_edge(invoc_name, arg_name)
    elif isinstance(arg, javalang.tree.MemberReference):
        arg_name = f"{invoc_name}.{arg.member}"
        tdg.add_node(arg_name, "MemberReference")
        tdg.add_edge(invoc_name, arg_name)
    elif isinstance(arg, javalang.tree.MethodInvocation):
        method_name = f"{invoc_name}.{arg.member}"
        tdg.add_node(method_name, "MethodInvocation")
        tdg.add_edge(invoc_name, method_name)
        if arg.arguments:
            for sub_arg in arg.arguments:
                process_method_argument(sub_arg, method_name, tdg)
    elif isinstance(arg, javalang.tree.ClassCreator):
        creator_type = get_type_name(arg.type)
        creator_name = f"{invoc_name}.{creator_type}"
        tdg.add_node(creator_name, "ClassCreator")
        tdg.add_edge(invoc_name, creator_name)

def process_project_directory(project_dir, output_json):
    tdg = TypeDependencyGraph()
    for root, dirs, files in os.walk(project_dir):
        for file in files:
            if file.endswith(".java"):
                file_path = os.path.join(root, file)
                process_java_file(file_path, tdg)
    tdg.save_to_json(output_json)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python create_tdg.py <project_directory>")
        sys.exit(1)

    project_directory = sys.argv[1]
    output_json = "type_dependency_graph.json"
    process_project_directory(project_directory, output_json)
    print(f"Type Dependency Graph saved to {output_json}")

