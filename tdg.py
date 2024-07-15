import os
import json
import sys
from collections import defaultdict
import javalang

def create_type_dependency_graph(java_project_dir):
    type_dependency_graph = defaultdict(lambda: {'dependencies': [], 'methods': []})

    for root, _, files in os.walk(java_project_dir):
        for file in files:
            if file.endswith('.java'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as java_file:
                    content = java_file.read()
                    tree = javalang.parse.parse(content)
                    type_dependency_graph = extract_dependencies(tree, type_dependency_graph)
    
    return type_dependency_graph

def extract_dependencies(tree, graph):
    current_class = None
    
    for _, node in tree:
        if isinstance(node, (javalang.tree.ClassDeclaration, javalang.tree.InterfaceDeclaration, javalang.tree.EnumDeclaration)):
            current_class = node.name
            if node.extends:
                if isinstance(node.extends, list):
                    for parent in node.extends:
                        graph[current_class]['dependencies'].append(parent.name)
                else:
                    graph[current_class]['dependencies'].append(node.extends.name)
            if node.implements:
                for implement in node.implements:
                    graph[current_class]['dependencies'].append(implement.name)
        elif isinstance(node, javalang.tree.MethodDeclaration) and current_class:
            current_method = node.name
            return_type = node.return_type.name if node.return_type else 'void'
            graph[current_class]['methods'].append({'method': current_method, 'return_type': return_type, 'parameters': [param.type.name for param in node.parameters]})
            if node.body:
                for path, sub_node in node:
                    if isinstance(sub_node, javalang.tree.MethodInvocation):
                        if sub_node.qualifier:
                            method_call = f"{sub_node.qualifier}.{sub_node.member}"
                        else:
                            method_call = sub_node.member
                        graph[current_class]['dependencies'].append(method_call)
        elif isinstance(node, javalang.tree.FieldDeclaration) and current_class:
            for declarator in node.declarators:
                field_type = node.type.name
                graph[current_class]['dependencies'].append(field_type)
    
    return graph

def save_graph_to_json(graph, output_file):
    with open(output_file, 'w') as f:
        json.dump(graph, f, indent=4)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_java_project>")
        sys.exit(1)

    java_project_directory = sys.argv[1]
    output_json_file = "type_dependency_graph.json"

    graph = create_type_dependency_graph(java_project_directory)
    save_graph_to_json(graph, output_json_file)

    print(f"Type dependency graph saved to {output_json_file}")
