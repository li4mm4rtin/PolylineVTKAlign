import os
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib.pyplot as plt
import numpy as np
import natsort


def angleBetweenVectors(v1, v2):
    """
    Calculates the angles between vectors
    :param v1: three-dimensional vector
    :param v2: three-dimensional vector
    :return: angle in radians between two vectors
    """
    return np.arccos(np.dot(v2, v1) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def rotatoChip(point, angle):
    """
    Rotates each point based on an angle. Assumes rotation about x-axis, to change you must change the rotation matrix
    to the correct one. Was hungry when this function was made, now I think it's funny, so it is staying.
    :param point: three-dimensional point
    :param angle: angle b
    :return: rotated three-dimensional point
    """
    rotationMatrixX = np.array([[1, 0, 0],
                                [0, np.cos(angle), -np.sin(angle)],
                                [0, np.sin(angle), np.cos(angle)]])

    return np.dot(point, rotationMatrixX)


# Define input and output directory for files
#### SHOULD ONLY NEED EDIT HERE ####
indirectory = '/Users/liammartin/Library/CloudStorage/OneDrive-UniversityofPittsburgh/Research/Projects/Cystocele/SSM/2D_SSM/1-SegmentedData/vtks_forFigure/'
outDirectory = None
outDirectory_notScaled = None

if outDirectory is not None and not os.path.isdir(outDirectory):
    os.mkdir(outDirectory)

# list all files in the input directory if they end with .vtk
filenames = natsort.natsorted(os.listdir(indirectory))
filenames = [filename for filename in filenames if filename.lower().endswith(".vtk")]

# Define chosen location to translate the polylines to.
origin = [0, 0, 0]

# Define scale term and vtk reader and writer
pclScaleLength = 0
reader = vtk.vtkPolyDataReader()
writer = vtk.vtkPolyDataWriter()

# loop through files
for filename in filenames:
    # loads vtk file
    reader.SetFileName(os.path.join(indirectory, filename))
    reader.Update()

    # gets output data from vtk file
    data = vtk_to_numpy(reader.GetOutput().GetPoints().GetData())

    # sets x-axis to zero and calculates flattened data
    data[:, 0] = 0
    flattenedDataLength = int(len(data) / 8) - 2

    # Defines variables needed for scale and rotation loop
    unscaled_data = new_data = np.zeros((flattenedDataLength, 3))
    j = 0
    cumulativeData = np.zeros(3)

    # Define vtk parameters
    vtk_points = vtk.vtkPoints()
    polyLine = vtk.vtkCellArray()
    polyLine.InsertNextCell(flattenedDataLength)

    # Loop through and compress data, this is an estimation of the center of the defined tube
    for i in range(len(data)):
        cumulativeData = cumulativeData + data[i]  # adds up 8 points
        if i > len(data) - 15:  # break loop before it loops back to the beginning
            break

        if (i + 1) % 8 == 0:  # average the 8 points to get the middle of the tube
            new_data[j, :] = cumulativeData / 8
            cumulativeData = np.zeros(3)
            j += 1

    # transform data to the origin
    transform = origin - new_data[0]
    if pclScaleLength == 0:  # calculates base scale term for the files
        pclVector = new_data[0] - new_data[-1]
        pclScaleLength = np.linalg.norm(pclVector)

    # transforms
    new_data[:, 1] += transform[1]
    new_data[:, 2] += transform[2]

    unscaled_data = new_data

    if outDirectory_notScaled is not None:
        for i in range(len(new_data)):
            vtk_points.InsertNextPoint(new_data[i])
            polyLine.InsertCellPoint(i)

        polyData = vtk.vtkPolyData()
        polyData.SetPoints(vtk_points)
        polyData.SetLines(polyLine)

        newFilename = os.path.basename(filename)[:-4] + '_flattenedTranslate.vtk'
        writer.SetFileName(os.path.join(outDirectory_notScaled, newFilename))
        writer.SetFileVersion(42)
        writer.SetInputData(polyData)
        writer.Write()

    # calculate difference between first and last point for scale and rotation
    pclVectorCurrent = new_data[0] - new_data[-1]
    print(f"{filename}: ", pclVectorCurrent)
    scale = pclScaleLength / np.linalg.norm(pclVectorCurrent)
    new_data = new_data * scale

    # calculates angle between the template shape and final shape
    angle_rads = angleBetweenVectors(pclVector, pclVectorCurrent)

    # checks if the rotation has to be positive or negative
    # this is a bad check, i should do this in the angle function
    if pclVector[2] > -1 * new_data[-1, 2]:
        angle_rads = -angle_rads

    # rotates the data and assembles the new polylines
    for i in range(len(new_data)):
        new_data[i] = rotatoChip(new_data[i], angle_rads)
        vtk_points.InsertNextPoint(new_data[i])
        polyLine.InsertCellPoint(i)

    # sets up the vtk data
    polyData = vtk.vtkPolyData()
    polyData.SetPoints(vtk_points)
    polyData.SetLines(polyLine)

    if outDirectory is not None:
        # writes the vtk files
        newFilename = os.path.basename(filename)[:-4] + '_aligned.vtk'
        writer.SetFileName(os.path.join(outDirectory, newFilename))
        writer.SetFileVersion(42)
        writer.SetInputData(polyData)
        writer.Write()

    if filename[0] == 'e':
        plt.figure('Scaled Data')
        plt.plot(new_data[:, 1], new_data[:, 2], color='r')
        pcl = [(0, 0), (new_data[-1, 1], new_data[-1, 2])]
        x, y = zip(*pcl)
        plt.plot(x, y, color='r', linestyle='dashed')
        plt.xticks([])
        plt.yticks([])

        plt.figure('Scaled Data - Evacuation')
        plt.plot(new_data[:, 1], new_data[:, 2], color='r')
        pcl = [(0, 0), (new_data[-1, 1], new_data[-1, 2])]
        x, y = zip(*pcl)
        plt.plot(x, y, color='r', linestyle='dashed')
        plt.xticks([])
        plt.yticks([])

        plt.figure('Unscaled Data')
        plt.plot(unscaled_data[:, 1], unscaled_data[:, 2], color='r')
        pcl = [(0, 0), (unscaled_data[-1, 1], unscaled_data[-1, 2])]
        x, y = zip(*pcl)
        plt.plot(x, y, color='r', linestyle='dashed')
        plt.xticks([])
        plt.yticks([])

        plt.figure('Unscaled Data - Evacuation')
        plt.plot(unscaled_data[:, 1], unscaled_data[:, 2], color='r')
        pcl = [(0, 0), (unscaled_data[-1, 1], unscaled_data[-1, 2])]
        x, y = zip(*pcl)
        plt.plot(x, y, color='r', linestyle='dashed')
        plt.xticks([])
        plt.yticks([])

    elif filename[0] == 'r':
        plt.figure('Scaled Data')
        plt.plot(new_data[:, 1], new_data[:, 2], color='b')
        pcl = [(0, 0), (new_data[-1, 1], new_data[-1, 2])]
        x, y = zip(*pcl)
        plt.plot(x, y, color='b', linestyle='dashed')

        plt.figure('Scaled Data - Rest')
        plt.plot(new_data[:, 1], new_data[:, 2], color='b')
        pcl = [(0, 0), (new_data[-1, 1], new_data[-1, 2])]
        x, y = zip(*pcl)
        plt.plot(x, y, color='b', linestyle='dashed')
        plt.xticks([])
        plt.yticks([])

        plt.figure('Unscaled Data')
        plt.plot(unscaled_data[:, 1], unscaled_data[:, 2], color='b')
        pcl = [(0, 0), (unscaled_data[-1, 1], unscaled_data[-1, 2])]
        x, y = zip(*pcl)
        plt.plot(x, y, color='b', linestyle='dashed')

        plt.figure('Unscaled Data - Rest')
        plt.plot(unscaled_data[:, 1], unscaled_data[:, 2], color='b')
        pcl = [(0, 0), (unscaled_data[-1, 1], unscaled_data[-1, 2])]
        x, y = zip(*pcl)
        plt.plot(x, y, color='b', linestyle='dashed')
        plt.xticks([])
        plt.yticks([])

    elif filename[0] == 's':
        plt.figure('Scaled Data')
        plt.plot(new_data[:, 1], new_data[:, 2], color='k')
        pcl = [(0, 0), (new_data[-1, 1], new_data[-1, 2])]
        x, y = zip(*pcl)
        plt.plot(x, y, color='k', linestyle='dashed')

        plt.figure('Scaled Data - Squeeze')
        plt.plot(new_data[:, 1], new_data[:, 2], color='k')
        pcl = [(0, 0), (new_data[-1, 1], new_data[-1, 2])]
        x, y = zip(*pcl)
        plt.plot(x, y, color='k', linestyle='dashed')
        plt.xticks([])
        plt.yticks([])

        plt.figure('Unscaled Data')
        plt.plot(unscaled_data[:, 1], unscaled_data[:, 2], color='k')
        pcl = [(0, 0), (unscaled_data[-1, 1], unscaled_data[-1, 2])]
        x, y = zip(*pcl)
        plt.plot(x, y, color='k', linestyle='dashed')

        plt.figure('Unscaled Data - Squeeze')
        plt.plot(unscaled_data[:, 1], unscaled_data[:, 2], color='k')
        pcl = [(0, 0), (unscaled_data[-1, 1], unscaled_data[-1, 2])]
        x, y = zip(*pcl)
        plt.plot(x, y, color='k', linestyle='dashed')
        plt.xticks([])
        plt.yticks([])

    else:
        print("BAD DATA")

    # # creates the plots
    # plt.figure('Scaled Data')
    # plt.plot(new_data[:, 1], new_data[:, 2])
    # pcl = [(0, 0), (new_data[-1, 1], new_data[-1, 2])]
    # x, y = zip(*pcl)
    # plt.plot(x, y)
    #
    # plt.figure('Unscaled Data')
    # plt.plot(unscaled_data[:, 1], unscaled_data[:, 2])
    # pcl = [(0, 0), (unscaled_data[-1, 1], unscaled_data[-1, 2])]
    # x, y = zip(*pcl)
    # plt.plot(x, y)

# plots data
plt.show()
