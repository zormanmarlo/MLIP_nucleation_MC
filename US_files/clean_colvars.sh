#!/bin/bash

# Loop through even numbers from 2 to 42
for ((i=2; i<=42; i+=2)); do
    dir="${i}mer"

    # Check if the directory exists
    if [ -d "$dir" ]; then
        # Handle multiple colvar files (if any)
        for file in "$dir"/colvar*; do
            # Check if the file exists and is a regular file
            if [ -f "$file" ]; then
                # Get total number of lines
                total_lines=$(wc -l < "$file")
                # Calculate halfway point (round up if odd)
                half_lines=$(( (total_lines + 1) / 2 ))

                # Create cleaned file
                tail -n +"$((half_lines + 1))" "$file" > "$dir/cleaned_colvar_$i.out"
            fi
        done
    else
        echo "Directory $dir does not exist, skipping."
    fi
done

