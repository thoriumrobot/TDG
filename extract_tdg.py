import os
import sys
import json
import javalang
import logging
import traceback
import networkx as nx
from tdg_utils import JavaTDG

def get_parent_id(file_name, parent):
    if parent is None:
        return None
    if hasattr(parent, 'name'):
        return f"{file_name}.{parent.name}"
    if isinstance(parent, javalang.tree.MethodInvocation):
        return f"{file_name}.{parent.member}"
    if isinstance(parent, javalang.tree.Assignment):
        if parent.position:
            return f"{file_name}.assignment_{parent.position.line}_{parent.position.column}"
    return None

def has_nullable_annotation(annotations):
    return any(annotation.name == 'Nullable' for annotation in annotations)

def get_actual_type(node):
    if hasattr(node, 'type') and hasattr(node.type, 'name'):
        return node.type.name
    return None

def process_file(file_path, output_dir):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        tree = javalang.parse.parse(content)
        file_name = os.path.basename(file_path)
        logging.info(f"Processing file {file_path}")

        tdg = JavaTDG()
        file_id = file_name
        tdg.add_node(file_id, "file", file_name)

        for path, node in tree:
            if isinstance(node, javalang.tree.ClassDeclaration):
                class_id = f"{file_name}.{node.name}"
                line_number = node.position.line if node.position else None
                tdg.add_node(class_id, "class", node.name, line_number=line_number)
                tdg.add_classname(node.name)
                tdg.add_edge(file_id, class_id, "contains")
                for method in node.methods:
                    method_id = f"{class_id}.{method.name}()"
                    line_number = method.position.line if method.position else None
                    nullable = has_nullable_annotation(method.annotations)
                    tdg.add_node(method_id, "method", method.name, line_number=line_number, nullable=nullable)
                    tdg.add_edge(class_id, method_id, "contains")
                    for param in method.parameters:
                        param_id = f"{method_id}.{param.name}"
                        line_number = param.position.line if param.position else None
                        nullable = has_nullable_annotation(param.annotations)
                        actual_type = get_actual_type(param)
                        tdg.add_node(param_id, "parameter", param.name, line_number=line_number, nullable=nullable, actual_type=actual_type)
                        tdg.add_edge(method_id, param_id, "has_parameter")
                for field in node.fields:
                    for decl in field.declarators:
                        field_id = f"{class_id}.{decl.name}"
                        line_number = decl.position.line if decl.position else None
                        nullable = has_nullable_annotation(field.annotations)
                        actual_type = get_actual_type(decl)
                        tdg.add_node(field_id, "field", decl.name, line_number=line_number, nullable=nullable, actual_type=actual_type)
                        tdg.add_edge(class_id, field_id, "has_field")
            elif isinstance(node, javalang.tree.MethodDeclaration):
                method_id = f"{file_name}.{node.name}()"
                line_number = node.position.line if node.position else None
                tdg.add_node(method_id, "method", node.name, line_number=line_number)
                for param in node.parameters:
                    param_id = f"{method_id}.{param.name}"
                    line_number = param.position.line if param.position else None
                    actual_type = get_actual_type(param)
                    tdg.add_node(param_id, "parameter", param.name, line_number=line_number, actual_type=actual_type)
                    tdg.add_edge(method_id, param_id, "has_parameter")
            elif isinstance(node, javalang.tree.FieldDeclaration):
                for decl in node.declarators:
                    field_id = f"{file_name}.{decl.name}"
                    line_number = decl.position.line if decl.position else None
                    actual_type = get_actual_type(decl)
                    tdg.add_node(field_id, "field", decl.name, line_number=line_number, actual_type=actual_type)
                    tdg.add_edge(file_name, field_id, "has_field")
            elif isinstance(node, javalang.tree.VariableDeclarator):
                var_id = f"{file_name}.{node.name}"
                line_number = node.position.line if node.position else None
                actual_type = get_actual_type(node)
                tdg.add_node(var_id, "variable", node.name, line_number=line_number, actual_type=actual_type)
            elif isinstance(node, javalang.tree.Literal) and node.value == "null":
                if node.position:
                    null_id = f"{file_name}.null_{node.position.line}_{node.position.column}"
                    tdg.add_node(null_id, "literal", "null", line_number=node.position.line)
                    parent = path[-2] if len(path) > 1 else None
                    parent_id = get_parent_id(file_name, parent)
                    if parent_id:
                        tdg.add_edge(parent_id, null_id, "contains")
        
        output_path = os.path.join(output_dir, f"{file_name}.json")
        save_tdg_to_json(tdg, output_path)
    except javalang.parser.JavaSyntaxError as e:
        logging.error(f"Syntax error in file {file_path}: {e}")
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")
        logging.error(traceback.format_exc())

def save_tdg_to_json(tdg, output_path):
    data = nx.node_link_data(tdg.graph)
    with open(output_path, 'w') as f:
        json.dump(data, f)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_tdg.py <ProjectDir> <OutputDir>")
        sys.exit(1)
    
    logging.basicConfig(level=logging.INFO)
    project_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    os.makedirs(output_dir, exist_ok=True)

    for root, _, files in os.walk(project_dir):
        for file in files:
            if file.endswith('.java'):
                process_file(os.path.join(root, file), output_dir)
