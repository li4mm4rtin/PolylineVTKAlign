import os
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib.pyplot as plt
import numpy as np
import natsort


def angleBetweenVectors(v1, v2):
    return np.arccos(
            np.dot(v2, v1) /
            (np.linalg.norm(v1) * np.linalg.norm(v2))
            )


def rotatoChip(point, angle):
    rotation_matrix = np.array([[1,             0,              0],
                                [0, np.cos(angle), -np.sin(angle)],
                                [0, np.sin(angle), np.cos(angle)]])

    return np.dot(point, rotation_matrix)


in_directory = './in_vtks/'
filenames = natsort.natsorted(os.listdir(in_directory))

filenames = [filename for filename in filenames if filename.lower().endswith(".vtk")]

origin = [0, 0, 0]

pclScaleLength = 0

for filename in filenames:
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(os.path.join(in_directory, filename))
    reader.Update()

    data = vtk_to_numpy(reader.GetOutput().GetPoints().GetData())

    data[:, 0] = 0
    flattenedDataLength = int(len(data) / 8) - 2

    new_data = np.empty((flattenedDataLength, 3))
    prerotated_new_data = np.empty((flattenedDataLength, 3))
    j = 0
    cumulativeData = np.zeros(3)

    vtk_points = vtk.vtkPoints()
    polyLine = vtk.vtkCellArray()
    polyLine.InsertNextCell(flattenedDataLength)

    for i in range(len(data)):
        cumulativeData = cumulativeData + data[i]
        if i > len(data) - 15:
            break
        if (i + 1) % 8 == 0:
            prerotated_new_data[j, :] = new_data[j, :] = cumulativeData / 8
            cumulativeData = np.zeros(3)
            j += 1

    transform = origin - new_data[0]
    if pclScaleLength == 0:
        pclVector = new_data[0] - new_data[-1]
        pclScaleLength = np.linalg.norm(pclVector)

    new_data[:, 1] += transform[1]
    new_data[:, 2] += transform[2]

    pclVectorCurrent = new_data[0] - new_data[-1]

    scale = pclScaleLength/np.linalg.norm(pclVectorCurrent)

    new_data = new_data * scale

    angle_rads = angleBetweenVectors(pclVector, pclVectorCurrent)

    if pclVector[2] > -1 * new_data[-1, 2]:
        angle_rads = -angle_rads

    for i in range(len(new_data)):
        new_data[i] = rotatoChip(new_data[i], angle_rads)
        vtk_points.InsertNextPoint(new_data[i])
        polyLine.InsertCellPoint(i)

    polyData = vtk.vtkPolyData()
    polyData.SetPoints(vtk_points)
    polyData.SetLines(polyLine)

    newFilename = os.path.basename(filename)[:-4] + '_aligned.vtk'

    outDirectory = './out_vtks/'

    if not os.path.isdir(outDirectory):
        os.mkdir(outDirectory)

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(os.path.join(outDirectory, newFilename))
    writer.SetFileVersion(42)
    writer.SetInputData(polyData)

    writer.Write()

    plt.plot(new_data[:, 1], new_data[:, 2])
    pcl = [(0, 0), (new_data[-1, 1], new_data[-1, 2])]
    x, y = zip(*pcl)
    plt.plot(x, y)

plt.show()
