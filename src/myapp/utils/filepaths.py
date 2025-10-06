import os

def resolve_project_path(relative_path: str) -> str:
    # __file__ here is the path of this utils file
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    return os.path.join(project_root, relative_path)
