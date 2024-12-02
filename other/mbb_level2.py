class Node:
    def __init__(self, level, path, bound, value):
        self.level = level  # Current level in tree
        self.path = path  # Current path/solution
        self.bound = bound  # Upper bound of possible solutions from this node
        self.value = value  # Current actual value
        self.promising = False  # Flag for most promising branch


class ModifiedBranchAndBound:
    def __init__(self, n, values):
        self.n = n  # Problem size
        self.values = (
            values  # Dictionary or function that gives values for combinations
        )
        self.best_value = float("-inf")
        self.best_solution = None
        self.nodes_at_level = {}  # Track nodes at each level

    def calculate_bound(self, node):
        """
        Calculate upper bound for node.
        This is problem-specific and should be modified based on your specific problem.
        """
        # Get remaining elements not in current path
        remaining_elements = [i for i in range(self.n) if i not in node.path]

        # Calculate potential from remaining single elements
        remaining_potential = 0
        for elem in remaining_elements:
            single_value = self.values.get(tuple([elem]), 0)
            remaining_potential += single_value

        # Return current value plus potential from remaining elements
        return node.value + remaining_potential

    def get_possible_extensions(self, current_path):
        """
        Get possible next elements that can be added to current path.
        """
        used = set(current_path)
        return [i for i in range(self.n) if i not in used]

    def find_most_promising_node(self, level):
        """
        Find the node with highest bound at given level
        """
        if level not in self.nodes_at_level:
            return None

        nodes = self.nodes_at_level[level]
        if not nodes:
            return None

        most_promising = max(nodes, key=lambda x: x.bound)
        most_promising.promising = True
        return most_promising

    def solve(self):
        # Initialize with root node
        root = Node(level=0, path=[], bound=float("inf"), value=0)
        queue = [root]
        self.nodes_at_level[0] = [root]

        while queue:
            current = queue.pop(0)

            # If this is beyond level 2 and not in the promising branch, skip
            if current.level > 2 and not current.promising:
                continue

            # Update best solution if current is better
            if current.value > self.best_value and len(current.path) > 0:
                self.best_value = current.value
                self.best_solution = current.path.copy()

            # Get possible next elements
            extensions = self.get_possible_extensions(current.path)

            # Create child nodes
            for next_elem in extensions:
                new_path = current.path + [next_elem]

                # Convert path to tuple for dictionary lookup
                path_tuple = tuple(sorted(new_path))
                new_value = self.values.get(path_tuple, 0)

                child = Node(
                    level=current.level + 1,
                    path=new_path,
                    value=new_value,
                    bound=float("inf"),
                )

                # Calculate bound
                child.bound = self.calculate_bound(child)

                # Only add to queue if bound is better than current best
                if child.bound > self.best_value:
                    queue.append(child)

                    # Track nodes at each level
                    if child.level not in self.nodes_at_level:
                        self.nodes_at_level[child.level] = []
                    self.nodes_at_level[child.level].append(child)

            # After level 2, identify most promising branch
            if current.level == 2:
                promising_node = self.find_most_promising_node(2)
                if promising_node:
                    # Clear non-promising nodes from queue
                    queue = [node for node in queue if node.promising]

        return self.best_solution, self.best_value


# Example usage
def example_usage():
    # Example problem: Finding valuable coalitions
    # Values dictionary represents the value of different combinations
    values = {
        (0,): 5,
        (1,): 4,
        (2,): 3,
        (0, 1): 10,
        (0, 2): 8,
        (1, 2): 7,
        (0, 1, 2): 15,
    }

    # Create solver instance
    solver = ModifiedBranchAndBound(n=3, values=values)

    # Solve the problem
    solution, value = solver.solve()

    print(f"Best solution found: {solution}")
    print(f"Value: {value}")

    # Print the promising branch that was fully explored
    for level, nodes in solver.nodes_at_level.items():
        promising_nodes = [node for node in nodes if node.promising]
        if promising_nodes:
            print(f"Level {level} promising node: {promising_nodes[0].path}")


if __name__ == "__main__":
    example_usage()
