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

def get_superclass_name(node):
    """
    Extracts the superclass name from a class declaration node, if present.
    """
    if node.extends:
        return node.extends.name
    return None

def process_field_declaration(field, class_id, tdg):
    """
    Processes field declarations and connects them to the TDG.
    """
    for decl in field.declarators:
        field_id = f"{class_id}.{decl.name}"
        line_number = field.position.line if field.position else None
        actual_type = get_actual_type(decl)
        nullable = has_nullable_annotation(field.annotations)
        
        # Add the field to the TDG
        tdg.add_node(field_id, "field", decl.name, line_number=line_number, actual_type=actual_type, nullable=nullable)
        tdg.add_edge(class_id, field_id, "has_field")

        # Handle assignment to field via method call (e.g., field initialization)
        if isinstance(decl.initializer, javalang.tree.MethodInvocation):
            method_call_id = f"{class_id}.{decl.initializer.member}()"
            tdg.add_edge(field_id, method_call_id, "assigned_from_method")

def process_method_invocation(method_id, class_id, method_invocation, tdg):
    """
    Handles method invocations, ensuring they are correctly linked to the TDG.
    """
    called_method_id = f"{class_id}.{method_invocation.member}()"
    tdg.add_edge(method_id, called_method_id, "calls")
    return called_method_id

def process_expression(expression, method_id, class_id, tdg):
    """
    Recursively processes expressions to extract method invocations and variable references.
    """
    # If the expression is a method invocation
    if isinstance(expression, javalang.tree.MethodInvocation):
        method_call_id = process_method_invocation(method_id, class_id, expression, tdg)
        return method_call_id

    # If the expression is a member reference (i.e., a variable)
    if isinstance(expression, javalang.tree.MemberReference):
        referenced_var_id = f"{method_id}.{expression.member}"
        return referenced_var_id

    # Recursively process binary operations (e.g., x + y)
    if isinstance(expression, javalang.tree.BinaryOperation):
        left_result = process_expression(expression.operandl, method_id, class_id, tdg)
        right_result = process_expression(expression.operandr, method_id, class_id, tdg)
        return left_result, right_result

    return None

def process_assignment(statement, method_id, class_id, tdg):
    """
    Processes assignments to variables, handling complex expressions involving method calls or other variables.
    """
    if isinstance(statement, javalang.tree.Assignment):
        assigned_var_id = f"{method_id}.{statement.left.name}"

        # Process the right-hand expression of the assignment (may contain method calls or variables)
        results = process_expression(statement.expression, method_id, class_id, tdg)
        
        # Handle results from expressions (could be method calls or variable references)
        if isinstance(results, tuple):  # If both left and right are processed (binary operations)
            for result in results:
                if result:
                    tdg.add_edge(assigned_var_id, result, "assigned_from_expression")
        elif results:
            tdg.add_edge(assigned_var_id, results, "assigned_from_expression")

        return assigned_var_id
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

                # Process each method in the class
                for method in node.methods:
                    method_id = f"{class_id}.{method.name}()"
                    line_number = method.position.line if method.position else None
                    nullable = has_nullable_annotation(method.annotations)
                    tdg.add_node(method_id, "method", method.name, line_number=line_number, nullable=nullable)
                    tdg.add_edge(class_id, method_id, "contains")

                    # Check for overridden methods (inheritance)
                    if any(annotation.name == "Override" for annotation in method.annotations):
                        superclass_name = get_superclass_name(node)
                        if superclass_name:
                            superclass_method_id = f"{superclass_name}.{method.name}()"
                            tdg.add_edge(method_id, superclass_method_id, "overrides")

                    # Add method return value as a node
                    return_id = f"{method_id}.return"
                    tdg.add_node(return_id, "return", "return_value", line_number=line_number)
                    tdg.add_edge(method_id, return_id, "has_return")

                    # Add method parameters and variables
                    for param in method.parameters:
                        param_id = f"{method_id}.{param.name}"
                        line_number = param.position.line if param.position else None
                        actual_type = get_actual_type(param)
                        nullable = has_nullable_annotation(param.annotations)
                        tdg.add_node(param_id, "parameter", param.name, line_number=line_number, actual_type=actual_type, nullable=nullable)
                        tdg.add_edge(method_id, param_id, "has_parameter")

                    # Process method body statements (assignments and standalone method calls)
                    for statement in method.body:
                        # Handle standalone method calls
                        if isinstance(statement, javalang.tree.MethodInvocation):
                            process_method_invocation(method_id, class_id, statement, tdg)

                        # Handle variable assignments (with method calls or variables)
                        process_assignment(statement, method_id, class_id, tdg)

                    # Add variables used in the method as nodes and connect them
                    for local_var in method.body:
                        if isinstance(local_var, javalang.tree.VariableDeclarator):
                            var_id = f"{method_id}.{local_var.name}"
                            line_number = local_var.position.line if local_var.position else None
                            actual_type = get_actual_type(local_var)
                            tdg.add_node(var_id, "variable", local_var.name, line_number=line_number, actual_type=actual_type)
                            tdg.add_edge(method_id, var_id, "has_variable")

                # Process field declarations
                for field in node.fields:
                    process_field_declaration(field, class_id, tdg)

            # Handle top-level method declarations
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

            # Handle field declarations at the top level
            elif isinstance(node, javalang.tree.FieldDeclaration):
                process_field_declaration(node, file_name, tdg)

            # Handle variables and null literals
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
