import sys

def build_tree(lines, only_dirs=False, only_py_files=False):
    tree = {}
    for line in lines:
        parts = line.strip().split('/')
        if only_py_files and not parts[-1].endswith('.py'):
            continue  # Skip non-Python files if filtering for .py files only
        node = tree
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        # Decide whether to add the last part of the path
        if not only_dirs:
            node[parts[-1]] = {}
        else:
            if '.' not in parts[-1]:  # Assume it's a directory if no extension
                node[parts[-1]] = {}
    return tree

def print_tree(node, prefix='', only_dirs=False):
    for i, (key, value) in enumerate(node.items()):
        connector = '|-- ' if i < len(node) - 1 else '`-- '
        if value or not only_dirs:  # It's a directory or we're showing all
            print(f'{prefix}{connector}{key}')
            print_tree(value, prefix + ('|   ' if i < len(node) - 1 else '    '), only_dirs)

if __name__ == '__main__':
    only_dirs = '--dirs' in sys.argv
    only_py_files = '--pyfiles' in sys.argv

    with open('files.txt', 'r') as f:
        lines = f.readlines()

    tree = build_tree(lines, only_dirs=only_dirs, only_py_files=only_py_files)
    print_tree(tree, only_dirs=only_dirs)
