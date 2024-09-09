import javalang

# Sample Java code
java_code = """
public class MyClass extends SuperClass {
    public void myMethod() {
        // Some method
    }
}
"""

# Parse the Java code
tree = javalang.parse.parse(java_code)

# Walk through the parsed nodes
for path, node in tree:
    if isinstance(node, javalang.tree.ClassDeclaration):
        print(f"Class: {node.name}")
        if node.extends:
            print(f"Extends: {node.extends.name}")
        else:
            print("No superclass")

