#!/bin/bash

# Check if path argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <directory_path>"
    exit 1
fi

# Get the target directory path
target_dir="$1"

# Check if the directory exists
if [ ! -d "$target_dir" ]; then
    echo "Error: Directory '$target_dir' does not exist"
    exit 1
fi

# Change to target directory
cd "$target_dir" || exit 1

# Count total directories
total_dirs=$(find "$target_dir" -maxdepth 1 -type d | wc -l)
# Subtract 1 to exclude the target directory itself
((total_dirs--))
current_dir=0
processed=0
skipped=0

echo "Found $total_dirs directories to process"
echo "----------------------------------------"

# Process each directory in the specified path
for dir in */; do
    if [ -d "$dir" ]; then
        ((current_dir++))
        echo "[$current_dir/$total_dirs] Processing directory: $dir"
        
        # Check if log file exists
        log_file=$(find "$dir" -name "global_waypoint_planner*.log" -type f)
        
        if [ -z "$log_file" ]; then
            echo "  Warning: No global_waypoint_planner*.log file found in $dir"
            ((skipped++))
            continue
        fi

        # Check if log file is readable
        if [ ! -r "$log_file" ]; then
            echo "  Error: Cannot read log file $log_file"
            ((skipped++))
            continue
        fi

        # Extract date and time from the last few lines
        last_line=$(tail -n 1 "$log_file" | grep -o '[0-9]\{4\}-[0-9]\{2\}-[0-9]\{2\} [0-9]\{2\}:[0-9]\{2\}:[0-9]\{2\}')
        second_last_line=$(tail -n 2 "$log_file" | head -n 1 | grep -o '[0-9]\{4\}-[0-9]\{2\}-[0-9]\{2\} [0-9]\{2\}:[0-9]\{2\}:[0-9]\{2\}')
        
        # Use the last line with datetime
        datetime=""
        if [ ! -z "$last_line" ]; then
            datetime="$last_line"
        elif [ ! -z "$second_last_line" ]; then
            datetime="$second_last_line"
        fi

        if [ ! -z "$datetime" ]; then
            # Format datetime for folder name (remove spaces and special chars)
            formatted_datetime=$(echo "$datetime" | tr -d ':-' | tr ' ' '_')
            
            # Create new directory name
            new_dirname="${formatted_datetime}_${dir%/}"
            
            # Rename directory if it doesn't already exist
            if [ ! -d "$new_dirname" ] && [ "$dir" != "$new_dirname/" ]; then
                mv "$dir" "$new_dirname"
                echo "  Success: Renamed '$dir' -> '$new_dirname'"
                ((processed++))
            else
                echo "  Warning: Directory '$new_dirname' already exists or no rename needed"
                ((skipped++))
            fi
        else
            echo "  Warning: No valid datetime found in log file"
            ((skipped++))
        fi
    fi
done

echo "----------------------------------------"
echo "Processing complete!"
echo "Total directories: $total_dirs"
echo "Successfully processed: $processed"
echo "Skipped: $skipped"
