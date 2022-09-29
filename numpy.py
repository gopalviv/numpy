import numpy as np
from PIL import Image
import matplotlib.image as img

# img = Image.open('page-analyzer.jpg')
# # convert image object into array
# imageToMatrice = np.asarray(img)

# printing shape of image
# print(imageToMatrice.shape)
# read an image
imageMat = img.imread('page-analyzer.jpg')
print("Image shape:", imageMat.shape)
# print(imageMat.shape[2])

# if image is colored (RGB)
if (imageMat.shape[2] == 3):
    # reshape it from 3D matrice to 2D matrice
    imageMat_reshape = imageMat.reshape(imageMat.shape[0],
                                        -1)
    print("Reshaping to 2D array:",
          imageMat_reshape.shape)



"""For a black and white or gray scale image: 
There is only one channel present, thus, the shape of the matrices would be (n, n) 
where n represents the dimension of the images (pixels), 
and values inside the matrix range from 0 to 255.
For color or RGB image: It will render a tensor of 3 channels, 
thus the shape of the matrices would be (n, n,3). Each channel is an (n, n) matrix 
where each entry represents respectively the level of Red, Green, or Blue at the actual location 
inside the image."""

"""Note: we can save only 1D or 2D matrix in a file, therefore, 
there would be no issue in the gray scale or black and white image as it is a 2D matrix, 
but we need to make sure that this works for a colored or RGB image, which is a 3D matrix."""

"""For plotting graphs in Python, we will use the Matplotlib library. Matplotlib is used along with NumPy 
data to plot any type of graph. From matplotlib we use the specific function i.e. pyplot(), 
which is used to plot two-dimensional data.

Different functions used are explained below:

np.arange(start, end): This function returns equally spaced values from the interval [start, end).
plt.title(): It is used to give a title to the graph. Title is passed as the parameter to this function.
plt.xlabel(): It sets the label name at X-axis. Name of X-axis is passed as argument to this function.
plt.ylabel(): It sets the label name at Y-axis. Name of Y-axis is passed as argument to this function.
plt.plot(): It plots the values of parameters passed to it together.
plt.show(): It shows all the graph to the console."""

import matplotlib.pyplot as plt

# data to be plotted
x = np.arange(1, 11)
y = np.array([100, 10, 300, 20, 500, 60, 700, 80, 900, 100])

# plotting
plt.title("Line graph")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.plot(x, y, color="green")
plt.show()

# We have used np.array() to convert a dictionary to nd array.
import numpy as np
from ast import literal_eval

# creating class of string
name_list = """{
"column0": {"First_Name": "Akash",
"Second_Name": "kumar", "Interest": "Coding"},

"column1": {"First_Name": "Ayush",
"Second_Name": "Sharma", "Interest": "Cricket"},

"column2": {"First_Name": "Diksha",
"Second_Name": "Sharma","Interest": "Reading"},

"column3": {"First_Name":" Priyanka",
"Second_Name": "Kumari", "Interest": "Dancing"}

}"""
print("Type of name_list created:\n",
      type(name_list))

# converting string type to dictionary
t = literal_eval(name_list)

# printing the original dictionary
print("\nPrinting the original Name_list dictionary:\n",
      t)

print("Type of original dictionary:\n",
      type(t))

# converting dictionary to numpy array
result_nparra = np.array([[v[j] for j in ['First_Name', 'Second_Name',
                                          'Interest']] for k, v in t.items()])

print("\nConverted ndarray from the Original dictionary:\n",
      result_nparra)

# printing the type of converted array
print("Type:\n", type(result_nparra))
