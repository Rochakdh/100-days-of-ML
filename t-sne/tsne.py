import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns #python visualization library based on matplotlib

from sklearn import datasets #ml library, rigid than tf and pytorch, less customizable
from sklearn import manifold #for non linear dimensionality reduction

data = datasets.fetch_openml(
    'mnist_784', #name of data set
    version=1, #version of data set
    return_X_y=True #if true return tuple (data,target),else return bunch object(dict. like objet)
)

pixel_values, targets = data  # (pandas dataframe, pandas series)
targets = targets.astype(int)

single_image = pixel_values.iloc[1, :].values.reshape(28, 28)
plt.imshow(single_image, cmap='gray')
plt.show()


tsne = manifold.TSNE(n_components=2, random_state=42)
#n_comopenent is Dimension of the embedded space, lower dimension that we want the high dimension data to be converted to

transformed_data = tsne.fit_transform(pixel_values.iloc[:3000, :])

tsne_df = pd.DataFrame(
    np.column_stack((transformed_data, targets[:3000])),
    columns=["x", "y", "targets"]
) # here x and y is are components after t-sne decomposition. Result is a dataframe
tsne_df.loc[:, "targets"] = tsne_df.targets.astype(int)
grid = sns.FacetGrid(tsne_df, hue="targets", height=40)

""" Hue variable as a third dimension along a depth axis, where different 
levels are plotted with different colors. Height represents the each facet height in inch"""

grid.map(plt.scatter, "x", "y").add_legend()
plt.show()