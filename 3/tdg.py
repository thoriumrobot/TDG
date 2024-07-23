import os
import json
import argparse
import javalang
import networkx as nx

def parse_java_file(file_path):
    with open(file_path, 'r') as file:
        return javalang.parse.parse(file.read())

def create_tdg(java_ast):
    tdg = nx.DiGraph()

    for path, node in java_ast:
        if isinstance(node, javalang.tree.MethodDeclaration):
            method_name = f"Method:{node.name}({', '.join(param.name for param in node.parameters)})"
            tdg.add_node(method_name, type='method', return_type=str(node.return_type))

            for param in node.parameters:
                param_name = f"Param:{param.name}"
                tdg.add_node(param_name, type='parameter', param_type=str(param.type))
                tdg.add_edge(param_name, method_name)

            if node.body:
                for statement in node.body:
                    if isinstance(statement, javalang.tree.VariableDeclaration):
                        for declarator in statement.declarators:
                            var_name = f"Var:{declarator.name}"
                            tdg.add_node(var_name, type='variable', var_type=str(statement.type))
                            tdg.add_edge(var_name, method_name)

        elif isinstance(node, javalang.tree.FieldDeclaration):
            for declarator in node.declarators:
                field_name = f"Field:{declarator.name}"
                tdg.add_node(field_name, type='field', field_type=str(node.type))

    return tdg

def save_tdg_as_json(tdg, output_file):
    tdg_dict = nx.readwrite.json_graph.node_link_data(tdg)
    with open(output_file, 'w') as file:
        json.dump(tdg_dict, file, indent=4)

def main(project_directory):
    tdg = nx.DiGraph()

    for root, _, files in os.walk(project_directory):
        for file in files:
            if file.endswith('.java'):
                file_path = os.path.join(root, file)
                try:
                    java_ast = parse_java_file(file_path)
                    file_tdg = create_tdg(java_ast)
                    tdg = nx.compose(tdg, file_tdg)
                except Exception as e:
                    print(f"Failed to parse {file_path}: {e}")

    output_file = os.path.join(project_directory, 'tdg.json')
    save_tdg_as_json(tdg, output_file)
    print(f"TDG saved as {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Type Dependency Graph for a Java Project")
    parser.add_argument('project_directory', type=str, help='Path to the Java project directory')
    args = parser.parse_args()
    main(args.project_directory)

