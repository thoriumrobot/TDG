import os
import sys
import json
import javalang
import networkx as nx
from collections import defaultdict
import logging

class JavaTDG:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.classnames = set()
        self.aliasgraph = nx.DiGraph()
        self.callgraph = defaultdict(list)

    def add_node(self, node_id, node_type, name):
        self.graph.add_node(node_id, type=node_type, name=name)

    def add_edge(self, from_node, to_node, edge_type):
        self.graph.add_edge(from_node, to_node, type=edge_type)

    def add_classname(self, classname):
        self.classnames.add(classname)

    def add_alias(self, from_node, to_node):
        self.aliasgraph.add_edge(from_node, to_node, type='alias')

    def to_json(self, file_path):
        data = {
            "nodes": [{"id": n, "attr": self.graph.nodes[n]} for n in self.graph.nodes()],
            "edges": [{"from": u, "to": v, "attr": self.graph.edges[u, v]} for u, v in self.graph.edges()],
            "classnames": list(self.classnames),
            "aliasgraph": [{"from": u, "to": v, "attr": self.aliasgraph.edges[u, v]} for u, v in self.aliasgraph.edges()],
            "callgraph": dict(self.callgraph)
        }
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

def alias_analysis(tdg, project_dir):
    logging.info("[1st Pass: Alias Analysis] Started...")
    for root, _, files in os.walk(project_dir):
        for file in files:
            if file.endswith('.java'):
                try:
                    with open(os.path.join(root, file), 'r') as f:
                        content = f.read()
                    tree = javalang.parse.parse(content)
                    file_name = os.path.basename(file)

                    # Process the AST nodes for alias analysis
                    for path, node in tree:
                        if isinstance(node, javalang.tree.VariableDeclarator):
                            var_name = node.name
                            if isinstance(node.initializer, javalang.tree.MemberReference):
                                alias_name = node.initializer.member
                                var_id = f"{file_name}.{var_name}"
                                alias_id = f"{file_name}.{alias_name}"
                                tdg.add_alias(alias_id, var_id)
                            elif isinstance(node.initializer, javalang.tree.Literal):
                                # Skip literals
                                pass
                            elif isinstance(node.initializer, javalang.tree.MethodInvocation):
                                # Skip method invocations
                                pass
                            else:
                                # Handle other initializers if necessary
                                pass
                except Exception as e:
                    logging.error(f"Error processing file {file}: {e}")
    logging.info("[1st Pass: Alias Analysis] Finished...")

def call_analysis(tdg, project_dir):
    logging.info("[2nd Pass: Call Analysis] Started...")
    for root, _, files in os.walk(project_dir):
        for file in files:
            if file.endswith(".java"):
                try:
                    with open(os.path.join(root, file), 'r') as f:
                        content = f.read()
                    tree = javalang.parse.parse(content)
                    for path, node in tree:
                        if isinstance(node, javalang.tree.MethodInvocation):
                            caller = ".".join(str(p) for p in path if isinstance(p, javalang.tree.MethodDeclaration))
                            callee = node.member
                            tdg.callgraph[caller].append(callee)
                except Exception as e:
                    logging.error(f"Error processing file {file}: {e}")
    logging.info(f"[2nd Pass: Call Analysis] Captured {sum(len(v) for v in tdg.callgraph.values())} call relationships.")
    logging.info("[2nd Pass: Call Analysis] Finished...")

def process_file(file_path, tdg):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        tree = javalang.parse.parse(content)
        file_name = os.path.basename(file_path)

        # Process the AST nodes
        for path, node in tree:
            if isinstance(node, javalang.tree.ClassDeclaration):
                class_id = f"{file_name}.{node.name}"
                tdg.add_node(class_id, "class", node.name)
                tdg.add_classname(node.name)
                for method in node.methods:
                    method_id = f"{class_id}.{method.name}()"
                    tdg.add_node(method_id, "method", method.name)
                    tdg.add_edge(class_id, method_id, "contains")
                    for param in method.parameters:
                        param_id = f"{method_id}.{param.name}"
                        tdg.add_node(param_id, "parameter", param.name)
                        tdg.add_edge(method_id, param_id, "has_parameter")
                for field in node.fields:
                    for decl in field.declarators:
                        field_id = f"{class_id}.{decl.name}"
                        tdg.add_node(field_id, "field", decl.name)
                        tdg.add_edge(class_id, field_id, "has_field")
            elif isinstance(node, javalang.tree.MethodDeclaration):
                method_id = f"{file_name}.{node.name}()"
                tdg.add_node(method_id, "method", node.name)
                for param in node.parameters:
                    param_id = f"{method_id}.{param.name}"
                    tdg.add_node(param_id, "parameter", param.name)
                    tdg.add_edge(method_id, param_id, "has_parameter")
            elif isinstance(node, javalang.tree.FieldDeclaration):
                for decl in node.declarators:
                    field_id = f"{file_name}.{decl.name}"
                    tdg.add_node(field_id, "field", decl.name)
                    tdg.add_edge(file_name, field_id, "has_field")
            elif isinstance(node, javalang.tree.VariableDeclarator):
                var_id = f"{file_name}.{node.name}"
                tdg.add_node(var_id, "variable", node.name)
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")

def process_directory(directory_path):
    tdg = JavaTDG()
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.java'):
                process_file(os.path.join(root, file), tdg)
    return tdg

def main():
    if len(sys.argv) != 2:
        print("Usage: python generate_tdg.py <JavaProjectDirectory>")
        sys.exit(1)

    logging.basicConfig(level=logging.INFO)

    directory_path = sys.argv[1]
    tdg = process_directory(directory_path)

    # Alias Analysis
    alias_analysis(tdg, directory_path)

    # Call Analysis
    call_analysis(tdg, directory_path)

    # Generate the final TDG
    tdg.to_json("tdg.json")
    logging.info("TDG saved to tdg.json")

if __name__ == "__main__":
    main()

