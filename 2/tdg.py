import os
import sys
import json
import javalang
import networkx as nx

def parse_java_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()
    tree = javalang.parse.parse(content)
    return tree

def extract_types(tree):
    types = {}
    for path, node in tree.filter(javalang.tree.TypeDeclaration):
        if isinstance(node, javalang.tree.ClassDeclaration) or isinstance(node, javalang.tree.InterfaceDeclaration):
            class_name = node.name
            types[class_name] = {
                'methods': {},
                'fields': {}
            }
            for member in node.body:
                if isinstance(member, javalang.tree.MethodDeclaration):
                    method_info = {
                        'return_type': str(member.return_type) if member.return_type else None,
                        'parameters': {param.name: str(param.type) for param in member.parameters},
                        'local_vars': {}
                    }
                    if member.body:
                        for mpath, mnode in member.body.filter(javalang.tree.LocalVariableDeclaration):
                            for decl in mnode.declarators:
                                method_info['local_vars'][decl.name] = str(mnode.type)
                    types[class_name]['methods'][member.name] = method_info
                elif isinstance(member, javalang.tree.FieldDeclaration):
                    for declarator in member.declarators:
                        types[class_name]['fields'][declarator.name] = str(member.type)
    return types

def build_tdg(types):
    graph = nx.DiGraph()

    for class_name, class_info in types.items():
        for method_name, method_info in class_info['methods'].items():
            method_node = f"{class_name}.{method_name}"
            if method_info['return_type']:
                graph.add_edge(method_node, method_info['return_type'], relation='returns')
            for param_name, param_type in method_info['parameters'].items():
                param_node = f"{class_name}.{method_name}.{param_name}"
                graph.add_edge(method_node, param_node, relation='parameter')
                graph.add_edge(param_node, param_type, relation='type')
            for var_name, var_type in method_info['local_vars'].items():
                var_node = f"{class_name}.{method_name}.{var_name}"
                graph.add_edge(method_node, var_node, relation='local_var')
                graph.add_edge(var_node, var_type, relation='type')

        for field_name, field_type in class_info['fields'].items():
            field_node = f"{class_name}.{field_name}"
            graph.add_edge(class_name, field_node, relation='field')
            graph.add_edge(field_node, field_type, relation='type')

    return graph

def save_graph(graph, output_filepath):
    data = nx.readwrite.json_graph.node_link_data(graph)
    with open(output_filepath, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def process_directory(directory):
    all_types = {}
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.java'):
                filepath = os.path.join(root, file)
                tree = parse_java_file(filepath)
                types = extract_types(tree)
                all_types.update(types)
    tdg = build_tdg(all_types)
    save_graph(tdg, os.path.join(directory, 'type_dependency_graph.json'))

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python create_tdg.py <path_to_java_project>")
        sys.exit(1)

    project_directory = sys.argv[1]
    if not os.path.isdir(project_directory):
        print(f"Error: {project_directory} is not a valid directory")
        sys.exit(1)

    process_directory(project_directory)
    print(f"Type dependency graph saved as type_dependency_graph.json in {project_directory}")

