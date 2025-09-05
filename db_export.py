import sqlite3
import json
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

def select_db_file():
    """
    Open file dialog to select rmc.db file
    Default directory is $HOME/maps/
    
    Returns:
        str: Selected file path or None if cancelled
    """
    # Hide the main tkinter window
    root = tk.Tk()
    root.withdraw()
    
    # Get default directory path
    default_dir = os.path.join(str(Path.home()), "maps")
    
    # Open file dialog
    file_path = filedialog.askopenfilename(
        initialdir=default_dir,
        title="Select rmc.db file",
        filetypes=[("Database files", "*.db"), ("All files", "*.*")],
        defaultextension=".db"
    )
    
    root.destroy()
    return file_path if file_path else None

def select_output_directory(initial_dir):
    """
    Open directory dialog to select output directory
    
    Args:
        initial_dir (str): Default directory path
    
    Returns:
        str: Selected directory path or None if cancelled
    """
    # Hide the main tkinter window
    root = tk.Tk()
    root.withdraw()
    
    # Open directory dialog
    dir_path = filedialog.askdirectory(
        initialdir=initial_dir,
        title="Select output directory"
    )
    
    root.destroy()
    return dir_path if dir_path else None

def export_tables_to_json(db_path, output_dir):
    """
    Export all tables from SQLite database to JSON files
    
    Args:
        db_path (str): Path to the SQLite database file
        output_dir (str): Directory to save JSON files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    exported_files = []  # Keep track of exported files
    
    # Export each table to JSON
    for table in tables:
        table_name = table[0]
        cursor.execute(f"SELECT * FROM {table_name}")
        
        # Get column names
        columns = [description[0] for description in cursor.description]
        
        # Fetch all rows
        rows = cursor.fetchall()
        
        # Convert to list of dictionaries
        table_data = []
        for row in rows:
            table_data.append(dict(zip(columns, row)))
        
        # Write to JSON file
        output_file = os.path.join(output_dir, f"{table_name}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(table_data, f, ensure_ascii=False, indent=2)
        
        exported_files.append(output_file)
            
    conn.close()
    return exported_files

if __name__ == "__main__":
    # Select database file using file dialog
    print("Please select the database file...")
    db_path = select_db_file()
    
    if db_path:
        # Get the directory of the selected database file
        default_output_dir = os.path.dirname(db_path)
        
        # Select output directory
        print("Please select the output directory...")
        output_dir = select_output_directory(default_output_dir)
        
        if output_dir:
            try:
                # Create json_rmc directory
                json_output_dir = os.path.join(output_dir, "json_rmc")
                
                # Check if directory exists and ask for confirmation
                if os.path.exists(json_output_dir):
                    root = tk.Tk()
                    root.withdraw()
                    if not messagebox.askyesno(
                        "Confirmation",
                        f"'{json_output_dir}' already exists.\nDo you want to overwrite it?"
                    ):
                        print("Operation cancelled.")
                        root.destroy()
                        exit()
                    root.destroy()
                
                exported_files = export_tables_to_json(db_path, json_output_dir)
                
                print(f"\nExport completed successfully.")
                print(f"Output directory: {json_output_dir}")
                print("\nExported files:")
                for file in exported_files:
                    print(f"- {os.path.basename(file)}")
                
            except Exception as e:
                print(f"An error occurred: {str(e)}")
        else:
            print("Output directory was not selected.")
    else:
        print("Database file was not selected.")
