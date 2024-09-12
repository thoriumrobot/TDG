import os
import re

# Specify the root directory where the Java files are located
root_dir = './reannotated'

# Regex pattern to match multiple occurrences of @Nullable
nullable_pattern = r'(@Nullable\s+)+'

def remove_duplicate_nullable(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Replace multiple occurrences of @Nullable with a single @Nullable
    updated_content = re.sub(nullable_pattern, '@Nullable ', content)
    
    # Write back the updated content to the file only if changes were made
    if content != updated_content:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(updated_content)
        print(f"Updated {file_path}")

def traverse_directory(directory):
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith('.java'):
                file_path = os.path.join(dirpath, filename)
                remove_duplicate_nullable(file_path)

if __name__ == '__main__':
    # Start traversing from the root directory
    traverse_directory(root_dir)

