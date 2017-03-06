import pandas as pd
from pathlib import Path
from scipy import misc
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
import glob
from sklearn import manifold

# Look pretty...

#img = misc.imread('D:\\learning\\DAT210x-master\\Module4\\Datasets\\ALOI\\32\\32_r0.png')

matplotlib.style.use('ggplot')

spl=glob.glob("D:\\learning\\DAT210x-master\\Module4\\Datasets\\ALOI\\32\*.png")
spll=glob.glob("D:\\learning\\DAT210x-master\\Module4\\Datasets\\ALOI\\32i\*.png")

#
# TODO: Start by creating a regular old, plain, "vanilla"
# python list. You can call it 'samples'.
#
# .. your code here .. 
samples=[]
colors=[]
# Open a file
#import os, sys

#path = "D:\\learning\\DAT210x-master\\Module4\\Datasets\\ALOI\\32"
#dirs = os.listdir( path )
#dirs

#
# TODO: Write a for-loop that iterates over the images in the
# Module4/Datasets/ALOI/32/ folder, appending each of them to
# your list. Each .PNG image should first be loaded into a
# temporary NDArray, just as shown in the Feature
# Representation reading.
#
# Optional: Resample the image down by a factor of two if you
# have a slower computer. You can also convert the image from
# 0-255  to  0.0-1.0  if you'd like, but that will have no
# effect on the algorithm's results.
#
# .. your code here .. 
for i in range(len(spl)):
    img=misc.imread(spl[i])
    img = img[::2, ::2]
    X = (img / 255.0).reshape(-1)
    samples.append(X)
    colors.append('b')
samples 

#path = "D:\learning\DAT210x-master\Module4\Datasets\ALOI\32"
#glob.glob('D:\learning\DAT210x-master\Module4\Datasets\ALOI\32\*.xlsx')


#
# TODO: Once you're done answering the first three questions,
# right before you converted your list to a dataframe, add in
# additional code which also appends to your list the images
# in the Module4/Datasets/ALOI/32_i directory. Re-run your
# assignment and answer the final question below.
#
# .. your code here .. 
for i in range(len(spll)):
    img=misc.imread(spll[i])
    img = img[::2, ::2]
    X = (img / 255.0).reshape(-1)
    samples.append(X)
    colors.append('r')
samples

#
# TODO: Convert the list to a dataframe
#
# .. your code here .. 
df = pd.DataFrame(samples)
df


#
# TODO: Implement Isomap here. Reduce the dataframe df down
# to three components, using K=6 for your neighborhood size
#
# .. your code here .. 
iso = manifold.Isomap(n_neighbors=6, n_components=3)
iso.fit(df)
manifold.Isomap(eigen_solver='auto', max_iter=None, n_components=2, n_neighbors=6,
    neighbors_algorithm='auto', path_method='auto', tol=0)
T = iso.transform(df)
ab=pd.DataFrame(T)



#
# TODO: Create a 2D Scatter plot to graph your manifold. You
# can use either 'o' or '.' as your marker. Graph the first two
# isomap components
#
# .. your code here .. 

ab.plot.scatter(x=0, y=1,c=colors)


#
# TODO: Create a 3D Scatter plot to graph your manifold. You
# can use either 'o' or '.' as your marker:
#
# .. your code here .. 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Final Grade')
ax.set_ylabel('First Grade')
ax.set_zlabel('Daily Alcohol')

ax.scatter(ab[0], ab[1],ab[2], c=colors, marker='.')
plt.show()

