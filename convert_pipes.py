import yaml
import json
import tkinter as tk
from tkinter import filedialog
import os
import re

def extract_yaml_from_shell_script(script_file):
    with open(script_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract YAML content from rosservice call command
    # Find the content after "receive_pipes" and before the closing quote
    yaml_match = re.search(r'receive_pipes\s+"(.*?)"', content, re.DOTALL)
    if yaml_match:
        yaml_content = yaml_match.group(1)
        # Convert the content to proper YAML format
        yaml_data = yaml.safe_load(yaml_content)  # Return YAML data directly
        return yaml_data
    else:
        raise ValueError("Could not find valid YAML content in the shell script")

# Convert YAML data to JSON files
def convert_yaml_to_json(yaml_data, script_file):
    # Get the directory of the shell script
    default_dir = os.path.dirname(os.path.abspath(script_file))
    
    # Select output directory
    save_dir = filedialog.askdirectory(
        title="Select output directory for JSON files",
        initialdir=default_dir
    )
    
    if not save_dir:
        print("No directory selected.")
        return
    
    # Process YAML data
    for i, pipe in enumerate(yaml_data['pipes']):
        new_pipe = {
            'id': i,
            'sector_id': 1,
            'location_id': 1,
            'name': pipe.get('name', f"Pipe {i}"),
            'closed': bool(pipe['closed']),
            'path': []
        }
        
        for j, checkpoint in enumerate(pipe['segments']):
            new_checkpoint = {
                'can_overtake': checkpoint.get('can_overtake', False),
                'id': j,
                'name': "",
                'point': checkpoint['point'],
                'radius': checkpoint.get('radius', 0),
                'shift_from_centre': checkpoint.get('shift_from_centre', 0)
            }
            new_pipe['path'].append(new_checkpoint)
        
        # Generate filename using pipe name
        filename = f"_{i:02d}_{pipe.get('name', f'Pipe_{i}')}.json"
        json_file = os.path.join(save_dir, filename)
        
        with open(json_file, 'w', encoding='utf-8') as jf:
            json.dump([new_pipe], jf, ensure_ascii=False, indent=2)
        print(f"Created: {json_file}")

# Main process
if __name__ == "__main__":
    # Create root window and hide it
    root = tk.Tk()
    root.withdraw()
    
    # Open file dialog
    home_dir = os.path.expanduser('~')  # Get home directory
    maps_dir = os.path.join(home_dir, 'maps')  # Create path to maps directory
    
    script_file = filedialog.askopenfilename(
        title="Select shell script file",
        initialdir=maps_dir,
        filetypes=[("Shell Script files", "*.sh"), ("All files", "*.*")]
    )
    
    if script_file:
        try:
            yaml_data = extract_yaml_from_shell_script(script_file)
            convert_yaml_to_json(yaml_data, script_file)
            print(f"Successfully converted {script_file} to JSON files.")
        except Exception as e:
            print(f"Error occurred: {str(e)}")
    else:
        print("No file selected.")
