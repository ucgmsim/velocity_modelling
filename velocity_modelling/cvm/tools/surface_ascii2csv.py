import sys
import csv
import numpy as np

def convert_to_csv(input_file, output_file):
    try:
        # Read the input file
        with open(input_file, 'r') as f:
            lines = f.readlines()

        # Parse grid dimensions
        grid_dims = lines[0].strip().split()
        nlat = int(grid_dims[0])  # 215 in your example
        nlon = int(grid_dims[1])  # 201 in your example

        # Parse latitude values (second line)
        lat_values = [float(x) for x in lines[1].strip().split()]

        # Parse longitude values (third line)
        lon_values = [float(x) for x in lines[2].strip().split()]

        # Parse depth data (remaining lines)
        depth_data = []
        for line in lines[3:]:
            depth_data.extend([float(x) for x in line.strip().split()])
        
        # Check if data length matches grid dimensions
        if len(depth_data) != nlat * nlon:
            raise ValueError(f"Data length ({len(depth_data)}) doesn't match grid dimensions ({nlat} x {nlon})")

        # Reshape data to match codebase (nlat, nlon) then transpose to (nlon, nlat)
        depth_array = np.array(depth_data).reshape((nlat, nlon)).T

        # Write to CSV
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(['Latitude', 'Longitude', 'Depth(m)'])

            # Write data rows - iterating with longitude as outer loop to match transposed array
            for i in range(nlat): 
                for j in range(nlon):  # longitude index
                 # latitude index
                    writer.writerow([
                        lat_values[i],
                        lon_values[j],
                        depth_array[j, i]  # Note the index order matches transposed array
                    ])

        print(f"Successfully converted {input_file} to {output_file}")

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
    except Exception as e:
        print(f"Error: {str(e)}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py input_file.txt")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = input_file.rsplit('.', 1)[0] + '.csv'
    
    convert_to_csv(input_file, output_file)

if __name__ == "__main__":
    main()