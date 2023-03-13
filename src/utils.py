import numpy as np
from scipy.ndimage import maximum_filter
import pandas as pd
from sklearn.neighbors import NearestNeighbors

import matplotlib.pyplot as plt
# create circular kernel:

def create_circular_kernel(r):
    """
    Creates a circular kernel of radius r.
    Parameters
    ----------
    r : int
        The radius of the kernel.

    Returns
    -------
    kernel : np.array
        A 2d array of the circular kernel.

    """
    
    span = np.linspace(-1,1,r*2)
    X,Y = np.meshgrid(span,span)
    return (X**2+Y**2)**0.5<=1

def get_kl_divergence(p,q):
    # mask = (p!=0) * (q!=0)
    output = np.zeros(p.shape)
    # output[mask] = p[mask]*np.log(p[mask]/q[mask])
    output[:] = p[:]*np.log(p[:]/q[:])
    return output

def determine_localmax(distribution, min_distance=3, min_expression=5):
    """
    Returns a list of local maxima in a kde of the data frame.
    Parameters
    ----------
    distribution : np.array
        A 2d array of the distribution.
    min_distance : int, optional
        The minimum distance between local maxima. The default is 3.
    min_expression : int, optional
        The minimum expression level to include in the histogram. The default is 5.

    Returns
    -------
    rois_x : list
        A list of x coordinates of local maxima.
    rois_y : list
        A list of y coordinates of local maxima.

    """
    localmax_kernel = create_circular_kernel(min_distance)
    localmax_projection = (distribution == maximum_filter(
        distribution, footprint=localmax_kernel))

    rois_x, rois_y = np.where((distribution > min_expression) & localmax_projection)

    return rois_x, rois_y, distribution[rois_x, rois_y]

## These functions are going to be seperated into a package of their own at some point:

from sklearn.decomposition import PCA as Dimred


def fill_color_axes(rgb,dimred=None):

    if dimred is None:
        dimred = Dimred(n_components=3)
        dimred.fit(rgb)

    facs = dimred.transform(rgb)

    # rotate the ica_facs 45 in all the dimensions:
    # define a 45-degree 3d rotation matrix 
    # (0.500 | 0.500 | -0.707
    # -0.146 | 0.854 | 0.500
    # 0.854 | -0.146 | 0.500)
    rotation_matrix = np.array([[0.500,0.500,-0.707],
                                [-0.146,0.854,0.500],
                                [0.854,-0.146,0.500]])

    # rotate the facs:
    facs = np.dot(facs,rotation_matrix)


    return facs,dimred


# create circular kernel:
def create_circular_kernel(kernel_width):
    span = np.linspace(-1,1,kernel_width)
    X,Y = np.meshgrid(span,span)
    return (X**2+Y**2)**0.5<=1


# normalize array:
def min_to_max(arr):
    arr=arr-arr.min(0,keepdims=True)
    arr/=arr.max(0,keepdims=True)
    return arr

# define a function that fits expression data to into the umap embeddings:
def transform_embeddings(expression,pca,embedder_2d,embedder_3d):

    factors = pca.transform(expression)

    embedding = embedder_2d.transform(factors)
    embedding_color = embedder_3d.transform(embedding)
    
    # embedding_color = (embedding_color-color_min)/(color_max-color_min)
    
    return embedding, embedding_color

# define a function that plots the embeddings, with celltype centers rendered as plt.texts on top:
def plot_embeddings(embedding,embedding_color,celltype_centers,celltypes):
    colors = np.clip(embedding_color.copy(),0,1)

    plt.scatter(embedding[:,0],embedding[:,1],c=(colors),alpha=0.1,marker='.')
    for i in range(len(celltypes)):
        plt.text(celltype_centers[i,0],celltype_centers[i,1],celltypes[i],color='k',fontsize=8)



# define a function that subsamples spots around x,y given a window size:
def get_spatial_subsample_mask(coordinate_df,x,y,plot_window_size=5):
    return (coordinate_df.x>x-plot_window_size)&(coordinate_df.x<x+plot_window_size)&(coordinate_df.y>y-plot_window_size)&(coordinate_df.y<y+plot_window_size)

# define a function that returns the k nearest neighbors of x,y:
def create_knn_graph(coords,k=10):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    return distances, indices

# get a kernel-weighted average of the expression values of the k nearest neighbors of x,y:
def get_knn_expression(distances,neighbor_indices,genes, gene_labels,bandwidth=2.5):

    weights = np.exp(-distances/bandwidth)
    local_expression = pd.DataFrame(index = genes, columns = np.arange(distances.shape[0])).astype(float)

    for i,gene in enumerate(genes):
        weights_ = weights.copy()
        weights_[(gene_labels[neighbor_indices])!=i] = 0
        local_expression.loc[gene,:] = weights_.sum(1)
    
    return local_expression

# def pixelmap_to_raw(x,y,):
#     shift_x = int((spot_df_raw.x/um_per_pixel).min())
#     shift_y = int((spot_df_raw.y/um_per_pixel).min())
#     return (x+shift_x)*um_per_pixel,(y+shift_y)*um_per_pixel

# def raw_to_pixelmap(x,y):
#     x = ((spot_df_raw.x/um_per_pixel).astype(int))
#     y = ((spot_df_raw.y/um_per_pixel).astype(int))

#     return (x/um_per_pixel-x.min()),(y/um_per_pixel-y.min())

# def get_celltype(expression_vector):
#     return celltypes[np.argmax(np.corrcoef(expression_vector,signatures.values.T)[0,1:])]

