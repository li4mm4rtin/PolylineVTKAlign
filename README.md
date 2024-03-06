# VTK Data Alignment and Visualization

## Overview
This Python script aligns and scales VTK polyline files. The code assumes that each of the files is a polyline file on the ZY-plane, serving as a preliminary step for a principal component analysis (statistical shape model). The first .vtk file in the specified folder is saved as the "template file," and all the other files are rotated and scaled to match this file. It would be trivial to change this code to accommodate alignment to another plane; just be sure to modify the rotation matrix in the rotatochip function.

Each file is saved in the output folder (out_vtks), which is created in the installation directory in case it does not already exist. This code was written as part of the methods for an unpublished paper. The repository will be updated when the manuscript is published with a link to it.

Seven example files demonstrating the code's functionality are included. Feel free to edit this code as needed for other projects.

## Requirements
- Python 3.x
- VTK
- NumPy
- Matplotlib
- natsort

Install the required Python packages using the following command:
```bash
pip install -r requirements.txt
```

## Usage

1. Point the script to the folder with your .vtk files
2. Run the script

The aligned vtk files are saved in ./out_vtks/ with the suffix "_aligned" by default

## Functions

`angleBetweenVectors(v1, v2)`
Calculates the angle between the two vectors in radians.

`rotatoChip(point, angle)`
Rotates each point passed in around the x-axis by the specified angle.

## Main Loop

1. Reads the .vtk's from specified directory
2. Translates the first point of the polylines to (0, 0)
3. Processes and aligns each of the polyLines based on the vector between the first and last point
4. Shows a plot of all the polylines as a check

## Notes

The author of this script only guarantee its accuracy for the specific use case outlined above. If a user of this script need to consider the positional data for their data this script will distort that. 

