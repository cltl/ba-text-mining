"""
This is a diy fix of an outdated eli5 library (v1.13), which eliminates the ImportError: cannot import name 'if_delegate_has_method' from 'sklearn.utils.metaestimators' error. It replaces occurences in two files - in case you find the "if_delegate_has_method" in more files, adjust the "files_to_update" variable accordingly (not tested).
Should be OS-invariant, tested on Windows only.
If needed, adjust the path to point to your eli5 library.
"""
import os
import re
from pathlib import Path

base_dir = Path.home() / "anaconda3/Lib/site-packages/eli5"


files_to_update = [
    base_dir / "lime/utils.py",
    base_dir / "sklearn/permutation_importance.py"
]


old_import = "from sklearn.utils.metaestimators import if_delegate_has_method"
new_import = "from sklearn.utils.metaestimators import available_if"
old_decorator = r"@if_delegate_has_method\(delegate='([^']+)'\)"
new_decorator = lambda match: f"@available_if(lambda self: hasattr(self, '{match.group(1)}'))"


def update_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()

        content = content.replace(old_import, new_import)

        content = re.sub(old_decorator, new_decorator, content)

        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(content)
        print(f"Updated: {filepath}")
    except Exception as e:
        print(f"Failed to update {filepath}: {e}")


for file_path in files_to_update:
    if file_path.exists():
        update_file(file_path)
    else:
        print(f"File not found: {file_path}")