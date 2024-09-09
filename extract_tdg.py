import os
import sys
import json
import javalang
import logging
import traceback
import networkx as nx
from tdg_utils import JavaTDG, has_nullable_annotation

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

                # Method declarations and connections
                for method in node.methods:
                    method_id = f"{class_id}.{method.name}()"
                    line_number = method.position.line if method.position else None
                    nullable = has_nullable_annotation(method.annotations)
                    tdg.add_node(method_id, "method", method.name, line_number=line_number, nullable=nullable)
                    tdg.add_edge(class_id, method_id, "contains")

                    # Handle overridden methods (inheritance)
                    if "override" in method.modifiers:
                        # Check if method is overriding a superclass method
                        method_id = f"{class_id}.{node.name}()"
                        superclass_method_id = f"{superclass_name}.{node.name}()"
                        tdg.add_edge(method_id, superclass_method_id, "overrides")

                    # Add method return value as a node
                    return_id = f"{method_id}.return"
                    tdg.add_node(return_id, "return", "return_value", line_number=line_number)
                    tdg.add_edge(method_id, return_id, "has_return")

                    # Add parameters as nodes and connect to the method
                    for param in method.parameters:
                        param_id = f"{method_id}.{param.name}"
                        line_number = param.position.line if param.position else None
                        actual_type = get_actual_type(param)
                        nullable = has_nullable_annotation(param.annotations)
                        tdg.add_node(param_id, "parameter", param.name, line_number=line_number, nullable=nullable, actual_type=actual_type)
                        tdg.add_edge(method_id, param_id, "has_parameter")

                # Handle field declarations and assignments
                for field in node.fields:
                    for decl in field.declarators:
                        field_id = f"{class_id}.{decl.name}"
                        line_number = field.position.line if field.position else None
                        actual_type = get_actual_type(decl)
                        nullable = has_nullable_annotation(field.annotations) #decl?
                        tdg.add_node(field_id, "field", decl.name, line_number=line_number, nullable=nullable, actual_type=actual_type)
                        tdg.add_edge(class_id, field_id, "has_field")

                    # Handle assignment to field via method call
                    if isinstance(decl.initializer, javalang.tree.MethodInvocation):
                        method_call_id = f"{class_id}.{decl.initializer.member}()"
                        tdg.add_edge(field_id, method_call_id, "assigned_from_method")

                    # Handle method calls and connect them
                    for statement in method.body:
                        if isinstance(statement, javalang.tree.MethodInvocation):
                            called_method_id = f"{class_id}.{statement.member}()"
                            tdg.add_edge(method_id, called_method_id, "calls")
                            # Reverse edge is automatically added elsewhere

                            # If method call is assigned to a variable
                            if isinstance(statement.parent, javalang.tree.Assignment):
                                assigned_var_id = f"{method_id}.{statement.parent.left.name}"
                                tdg.add_edge(assigned_var_id, called_method_id, "assigned_from_method")

                    # Add variables used in the method as nodes and connect them
                    for local_var in method.body:
                        if isinstance(local_var, javalang.tree.VariableDeclarator):
                            var_id = f"{method_id}.{local_var.name}"
                            line_number = local_var.position.line if local_var.position else None
                            actual_type = get_actual_type(local_var)
                            tdg.add_node(var_id, "variable", local_var.name, line_number=line_number, actual_type=actual_type)
                            tdg.add_edge(method_id, var_id, "has_variable")

                    # Handle method calls and connect them
                    for statement in method.body:
                        if isinstance(statement, javalang.tree.MethodInvocation):
                            called_method_id = f"{class_id}.{statement.member}()"
                            if called_method_id in tdg.graph.nodes:
                                tdg.add_edge(method_id, called_method_id, "calls")

                    # Handle variable assignments through method calls
                        if isinstance(statement, javalang.tree.Assignment) and isinstance(statement.expression, javalang.tree.MethodInvocation):
                            assigned_var_id = f"{method_id}.{statement.expression.member}"
                            method_call_id = f"{class_id}.{statement.expression.member}()"
                            if assigned_var_id in tdg.graph.nodes:
                                tdg.add_edge(assigned_var_id, method_call_id, "assigned_through_method_call")

                    # Handle assignments to variables (literal nulls)
                    for statement in method.body:
                        if isinstance(statement, javalang.tree.Assignment):
                            if isinstance(statement.expression, javalang.tree.Literal) and statement.expression.value == "null":
                                assigned_var_id = f"{method_id}.{statement.left.name}"
                                tdg.add_edge(assigned_var_id, "null", "assigned_null")

            # Handle field declarations and assignments
            elif isinstance(node, javalang.tree.FieldDeclaration):
                for decl in node.declarators:
                    field_id = f"{file_name}.{decl.name}"
                    line_number = field.position.line if field.position else None
                    actual_type = get_actual_type(decl)
                    nullable = has_nullable_annotation(node.annotations)
                    tdg.add_node(field_id, "field", decl.name, line_number=line_number, actual_type=actual_type, nullable=nullable)
                    tdg.add_edge(file_name, field_id, "has_field")

                    # Handle assignment to field via method call
                    if isinstance(decl.initializer, javalang.tree.MethodInvocation):
                        method_call_id = f"{file_name}.{decl.initializer.member}()"
                        tdg.add_edge(field_id, method_call_id, "assigned_from_method")

            elif isinstance(node, javalang.tree.MethodDeclaration):
                method_id = f"{file_name}.{node.name}()"
                line_number = node.position.line if node.position else None
                tdg.add_node(method_id, "method", node.name, line_number=line_number)
                for param in node.parameters:
                    param_id = f"{method_id}.{param.name}"
                    line_number = param.position.line if param.position else None
                    actual_type = get_actual_type(param)
                    nullable = has_nullable_annotation(param.annotations)
                    tdg.add_node(param_id, "parameter", param.name, line_number=line_number, actual_type=actual_type, nullable=nullable)
                    tdg.add_edge(method_id, param_id, "has_parameter")
            elif isinstance(node, javalang.tree.FieldDeclaration):
                for decl in node.declarators:
                    field_id = f"{file_name}.{decl.name}"
                    line_number = field.position.line if decl.position else None
                    actual_type = get_actual_type(decl)
                    nullable = has_nullable_annotation(node.annotations) #decl? field?
                    tdg.add_node(field_id, "field", decl.name, line_number=line_number, actual_type=actual_type, nullable=nullable)
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
