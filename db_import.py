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

def select_json_directory(initial_dir):
    """
    Open directory dialog to select json directory
    
    Args:
        initial_dir (str): Default directory path (same as db file directory)
    
    Returns:
        str: Selected directory path or None if cancelled
    """
    # Hide the main tkinter window
    root = tk.Tk()
    root.withdraw()
    
    # Open directory dialog with mustexist option
    dir_path = filedialog.askdirectory(
        initialdir=initial_dir,
        title="Select JSON directory",
        mustexist=True  # Directory must exist
    )
    
    root.destroy()
    return dir_path if dir_path else None

def import_json_to_db(db_path, json_dir):
    """
    Import JSON files to SQLite database
    
    Args:
        db_path (str): Path to the SQLite database file
        json_dir (str): Directory containing JSON files
    """
    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    imported_tables = []
    
    # SQLite keywords list
    sqlite_keywords = {
        'group', 'order', 'table', 'index', 'primary', 'unique', 
        'key', 'where', 'from', 'select', 'update', 'insert', 
        'delete', 'drop', 'create', 'alter'
    }
    
    # Get list of JSON files
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    
    for json_file in json_files:
        table_name = os.path.splitext(json_file)[0]
        file_path = os.path.join(json_dir, json_file)
        
        try:
            # Read JSON data
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if not data:  # Skip empty files
                continue
                
            # Get column names from the first record
            columns = list(data[0].keys())
            
            # Check if table exists
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
            table_exists = cursor.fetchone() is not None
            
            if table_exists:
                # Delete existing data
                cursor.execute(f"DELETE FROM {table_name}")
                # Reset sequence for this table if it exists in sqlite_sequence
                cursor.execute("DELETE FROM sqlite_sequence WHERE name=?", (table_name,))
            else:
                # Create table if it doesn't exist
                # Escape column names if they are SQLite keywords
                columns_sql = ', '.join([
                    f"`{col}` TEXT" if col.lower() in sqlite_keywords else f"{col} TEXT"
                    for col in columns
                ])
                cursor.execute(f"CREATE TABLE {table_name} ({columns_sql})")
            
            # Prepare INSERT statement with escaped column names
            escaped_columns = [
                f"`{col}`" if col.lower() in sqlite_keywords else col
                for col in columns
            ]
            placeholders = ', '.join(['?' for _ in columns])
            insert_sql = f"INSERT INTO {table_name} ({', '.join(escaped_columns)}) VALUES ({placeholders})"
            
            # Insert data
            for row in data:
                values = [row.get(col) for col in columns]
                cursor.execute(insert_sql, values)
            
            imported_tables.append(table_name)
            
        except Exception as e:
            print(f"Error occurred while importing table '{table_name}': {str(e)}")
            print(f"Error details: {e.__class__.__name__}")
            continue
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    
    return imported_tables

if __name__ == "__main__":
    # Select database file
    print("Please select the database file...")
    db_path = select_db_file()
    
    if db_path:
        # Get the directory of the selected database file
        db_dir = os.path.dirname(db_path)
        
        # Select JSON directory with default path same as db file
        print("Please select the JSON directory...")
        json_dir = select_json_directory(db_dir)
        
        if json_dir:
            try:
                # Confirm before proceeding
                root = tk.Tk()
                root.withdraw()
                if messagebox.askyesno(
                    "Confirmation",
                    "Existing tables in the database will be overwritten.\nDo you want to continue?"
                ):
                    imported_tables = import_json_to_db(db_path, json_dir)
                    
                    print(f"\nImport completed successfully.")
                    print("\nImported tables:")
                    for table in imported_tables:
                        print(f"- {table}")
                else:
                    print("Operation cancelled.")
                root.destroy()
                
            except Exception as e:
                print(f"An error occurred: {str(e)}")
        else:
            print("JSON directory was not selected.")
    else:
        print("Database file was not selected.")
