import os
import re


def find_imports_in_file(file_path):
    with open(file_path, encoding="utf-8") as file:
        content = file.read()
    imports = re.findall(
        r"^\s*(?:import|from)\s+([a-zA-Z0-9_\.]+)", content, re.MULTILINE
    )
    return imports


def find_imports_in_directory(directory):
    imports = set()
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                imports.update(find_imports_in_file(file_path))
    return imports


def main():
    project_directory = "."  # Change this to your project directory
    imports = find_imports_in_directory(project_directory)
    print("Identified packages:")
    for imp in sorted(imports):
        print(imp)


if __name__ == "__main__":
    main()
