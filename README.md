# VTK Data Alignment and Visualization

## Overview
This Python script aligns and scales VTK polyline files, this code assumes that each of the files are polyline files on the ZY-plane. The first .vtk file in the specified folder is saved as the "template file" and all of the other files are rotated and scaled to match this file. It would be trivail to change this code to accomidate alignment to another plane, just be sure to change the rotation matrix in the function rotatochip. 

Each file is saved in the output folder

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


