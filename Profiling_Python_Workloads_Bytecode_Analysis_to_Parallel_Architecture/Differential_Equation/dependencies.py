import ast
import sys

class LoopDependencyAnalyzer(ast.NodeVisitor):
    def __init__(self):
        # List of tuples: (line_number, message)
        self.parallel_candidates = []

    def visit_For(self, node):
        # Extract the loop variable(s) used in the for loop.
        loop_target_vars = self.get_target_names(node.target)
        assigned_vars = set()
        side_effects = False

        # Walk through all nodes inside the loop body.
        for child in ast.walk(node):
            # Check for normal assignments.
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    names = self.get_target_names(target)
                    # If a name is assigned that's not the loop variable,
                    # then we mark it as a dependency.
                    for n in names:
                        if n not in loop_target_vars:
                            assigned_vars.add(n)
            # Check for augmented assignments (like x += 1).
            elif isinstance(child, ast.AugAssign):
                names = self.get_target_names(child.target)
                for n in names:
                    if n not in loop_target_vars:
                        assigned_vars.add(n)
            # Function calls are assumed to possibly have side effects.
            elif isinstance(child, ast.Call):
                side_effects = True

        # If no extra assignments or side effects were detected, consider this loop
        # as a potential candidate for parallel execution.
        if not assigned_vars and not side_effects:
            self.parallel_candidates.append((node.lineno, "For loop may be parallelizable"))
        
        # Continue analyzing inner nodes.
        self.generic_visit(node)

    def get_target_names(self, node):
        """
        Recursively extract variable names from assignment targets.
        """
        names = []
        if isinstance(node, ast.Name):
            names.append(node.id)
        elif isinstance(node, (ast.Tuple, ast.List)):
            for elt in node.elts:
                names.extend(self.get_target_names(elt))
        elif isinstance(node, ast.Attribute):
            # Attributes (like obj.attr) are considered non-local for this simple analysis.
            names.append(node.attr)
        return names

def analyze_file(filename):
    """
    Parse the file and analyze its AST for potential parallelizable loops.
    Returns a list of tuples containing the line number and a message.
    """
    with open(filename, "r") as f:
        source = f.read()
    tree = ast.parse(source, filename)
    analyzer = LoopDependencyAnalyzer()
    analyzer.visit(tree)
    return analyzer.parallel_candidates

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_parallelism.py <python_file>")
        sys.exit(1)
    filename = sys.argv[1]
    candidates = analyze_file(filename)
    if candidates:
        print("Potential parallelizable loops found:")
        for lineno, message in candidates:
            print(f"Line {lineno}: {message}")
    else:
        print("No clear parallelizable loops found.")