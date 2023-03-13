import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import numba

from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA as dim_reduction
import umap 

from src.utils import *

# This is a package to detect overlapping cells in a 2d spatial transcriptomics sample.


def assign_xy(df, xy_columns=['x', 'y'], grid_size=1):
    """
    Assigns an x,y coordinate to a pd.DataFrame of coordinates.
    Parameters
    ----------
    df : pd.DataFrame
        A dataframe of coordinates.
    xyz_columns : list, optional
        The names of the columns containing the x,y,z coordinates. The default is ['x','y', 'z'].
    grid_size : int, optional
        The size of the grid. The default is 1.

    Returns
    -------
    df : pd.DataFrame
        A dataframe with an x,y coordinate assigned to each row.

    """
    df['x_pixel'] = (df[xy_columns[0]] / grid_size).astype(int)
    df['y_pixel'] = (df[xy_columns[1]] / grid_size).astype(int)

    # assign each pixel a unique id
    df['n_pixel'] = df['x_pixel'] + df['y_pixel'] * df['x_pixel'].max()

    return df

def assign_z_median(df, z_column='z'):
    """
    Assigns a z coordinate to a pd.DataFrame of coordinates.
    Parameters
    ----------
    df : pd.DataFrame
        A dataframe of coordinates.
    z_column : str, optional
        The name of the column containing the z coordinate. The default is 'z'.

    Returns
    -------
    df : pd.DataFrame
        A dataframe with a z coordinate assigned to each row.

    """
    if not 'n_pixel' in df.columns:
        print(
            'Please assign x,y coordinates to the dataframe first by running assign_xy(df)')
    medians = df.groupby('n_pixel')[z_column].median()
    df['z_delim'] = medians[df.n_pixel].values

    return medians


def assign_z_mean(df, z_column='z'):
    """
    Assigns a z coordinate to a pd.DataFrame of coordinates.
    Parameters
    ----------
    df : pd.DataFrame
        A dataframe of coordinates.
    z_column : str, optional
        The name of the column containing the z coordinate. The default is 'z'.

    Returns
    -------
    df : pd.DataFrame
        A dataframe with a z coordinate assigned to each row.

    """
    if not 'n_pixel' in df.columns:
        print(
            'Please assign x,y coordinates to the dataframe first by running assign_xy(df)')
    means = df.groupby('n_pixel')[z_column].mean()
    df['z_delim'] = means[df.n_pixel].values

    return means


def create_histogram(df, genes=None, min_expression=0, KDE_bandwidth=None, grid_size=1):
    """
    Creates a 2d histogram of the data frame's [x,y] coordinates.
    Parameters
    ----------
    df : pd.DataFrame
        A dataframe of coordinates.
    genes : list, optional
        A list of genes to include in the histogram. The default is None.
    min_expression : int, optional
        The minimum expression level to include in the histogram. The default is 5.
    KDE_bandwidth : int, optional
        The bandwidth of the gaussian blur applied to the histogram. The default is 1.
    grid_size : int, optional
        The size of the grid. The default is 1.

    Returns
    -------
    hist : np.array
        A 2d array of the histogram.

    """
    if genes is None:
        genes = df['gene'].unique()

    x_max = df['x_pixel'].max()
    y_max = df['y_pixel'].max()

    df = df[df['gene'].isin(genes)]

    hist, xedges, yedges = np.histogram2d(df['x_pixel'], df['y_pixel'],
                                          bins=[np.arange(x_max+2),
                                                np.arange(x_max+2)])

    if KDE_bandwidth is not None:
        hist = gaussian_filter(hist, sigma=KDE_bandwidth)

    hist[hist < min_expression] = 0

    return hist





def get_rois(df, genes=None, min_distance=10, KDE_bandwidth=1, min_expression=5):
    """
    Returns a list of local maxima in a kde of the data frame.
    Parameters
    ----------
    df : pd.DataFrame
        A dataframe of coordinates.
    min_distance : int
        The minimum distance between local maxima.

    Returns
    -------
    rois : list
        A list of local maxima in a kde of the data frame.

    """

    if genes is None:
        genes = sorted(df.gene.unique())

    hist = create_histogram(
        df, genes=genes, min_expression=min_expression, KDE_bandwidth=KDE_bandwidth)

    rois_x, rois_y, _ = determine_localmax(
        hist, min_distance=min_distance, min_expression=min_expression)

    # rois_n_pixel = rois_x+rois_y*df.x_pixel.max()

    # if 'z_delim' in df.columns:
    #     c = pd.Series(index=rois_n_pixel)
    #     c[:] = 0
    #     c_ = df.n_pixel[df.n_pixel.isin(rois_n_pixel)].groupby(
    #         df.n_pixel).apply(lambda x: len(x))
    #     c[c_.index] = c_

    # else:
    #     c = 'r'

    return rois_x, rois_y

def get_expression_vectors_at_rois(df,rois_x, rois_y,genes = None, KDE_bandwidth= 1, min_expression = 0):
    """
    Returns a matrix of gene expression vectors at each local maximum.
    Parameters
    ----------
    df : pd.DataFrame
        A dataframe of coordinates.
    kde_plot_window_size : int
    Returns
    -------
    """

    if genes is None:
        genes = sorted(df.gene.unique())

    rois_n_pixel = rois_x+rois_y*df.x_pixel.max()

    expressions = pd.DataFrame(index=genes, columns=rois_n_pixel,dtype=float)
    expressions[:] = 0

    # print(expressions)

    for gene in genes:
        hist = create_histogram(df, genes=[gene], min_expression=min_expression, KDE_bandwidth=KDE_bandwidth)

        expressions.loc[gene] = hist[rois_x, rois_y]

    return expressions

def compute_divergence(df, genes, KDE_bandwidth=1, threshold_fraction=0.5, min_distance=3, min_expression=5, density_weight=2,  plot=False):
    """
    Computes the divergence between the top and bottom of the cell.
    Parameters
    ----------
    df : pd.DataFrame
        A dataframe of coordinates.
    genes : list
        A list of genes to compute the divergence for.
    KDE_bandwidth : int
        The bandwidth of the KDE.
    threshold_fraction : float
        The fraction of the loss score's maximum, used as a cutoff value.
    min_distance : int
        The minimum distance between two retrieved regions of interest.
    plot : bool
        Whether to plot the KDE.
    Returns
    -------
    divergence : np.array
        A matrix of divergence values.
    """

    hist_sum = create_histogram(
        df, genes=genes, min_expression=min_expression, KDE_bandwidth=KDE_bandwidth)

    divergence = np.zeros_like(hist_sum)

    df_top = df[df.z_delim < df.z]
    df_bottom = df[df.z_delim > df.z]

    for gene in genes:

        hist_top = create_histogram(
            df_top, genes=[gene], min_expression=0, KDE_bandwidth=KDE_bandwidth,)
        hist_bottom = create_histogram(
            df_bottom, genes=[gene], min_expression=0, KDE_bandwidth=KDE_bandwidth,)

        mask = (hist_top > 0) & (hist_bottom > 0) & (hist_sum > 0)
        hist_top[mask] /= hist_sum[mask]
        hist_bottom[mask] /= hist_sum[mask]

        divergence[mask] += get_kl_divergence(
            hist_top[mask], hist_bottom[mask])
        divergence[mask] += get_kl_divergence(
            hist_bottom[mask], hist_top[mask])

    distance_score = divergence*hist_sum*density_weight
    distance_threshold = distance_score.max()*threshold_fraction

    rois_x, rois_y, distance_score = determine_localmax(distance_score, min_distance, distance_threshold)

    if plot:
        plt.imshow(hist_sum, cmap='Greens')
        alpha = np.nan_to_num(divergence)
        alpha = alpha - alpha.min()
        alpha = alpha/alpha.max()

        plt.imshow(divergence, cmap='Reds', alpha=alpha**0.5)
        plt.scatter(rois_y, rois_x, c='b', marker='x')

    return rois_x, rois_y, distance_score

def find_overlaps(coordinate_df=None,
                  adata=None, 
                  coordinates_key='spatial',
                  genes_key='gene',
                  genes=None,
                  KDE_bandwidth=1.0,
                  threshold_fraction=0.5,
                  min_distance=10,):
    """
    Finds regions of overlap between the top and bottom of the tissue sample.
    Parameters
    ----------
    coordinate_df : pd.DataFrame
        A dataframe of coordinates.
    adata : anndata.AnnData
        An AnnData object containing the coordinates.
    coordinates_key : str
        The key in the AnnData object's uns attribute containing the coordinates.
    genes_key : str
        The key in the AnnData object's uns attribute containing the genes.
    genes : list
        A list of genes to compute the divergence for.
    KDE_bandwidth : float
        The bandwidth of the KDE.
    threshold_fraction : float
        The fraction of the divergence score's maximum, used as a cutoff value.
    
    """	

    if (coordinate_df is None) and (adata is None):
        raise ValueError('Either adata or coordinate_df must be provided.')
    

    if coordinate_df is None:
        coordinate_df = adata.uns[coordinates_key]

    if genes is None:
        genes = sorted(coordinate_df[genes_key].unique())

    assign_xy(coordinate_df)
    assign_z_mean(coordinate_df)

    rois_x, rois_y, divergence = compute_divergence(coordinate_df, 
                            genes, 
                            KDE_bandwidth=KDE_bandwidth, 
                            threshold_fraction=threshold_fraction,
                            min_distance=min_distance)
    
    if adata is not None:
        adata.uns['rois'] = pd.DataFrame({'x':rois_x, 'y':rois_y, 'divergence':divergence})
        return adata.uns['rois']
    else:
        return pd.DataFrame({'x':rois_x, 'y':rois_y, 'divergence':divergence})
    

def visualize_rois(coordinate_df=None,
                   roi_df=None,
                  adata=None, 
                  n_cases=3,
                  genes=None,
                  gene_key='gene',
                  signature_matrix=None,
                  coordinates_key='spatial',
                  KDE_bandwidth=1.5,
                  celltyping_min_expression=10,
                  celltyping_min_distance=5,
                  plot_window_size=30):
    """
    """

    if (coordinate_df is None) and (adata is None):
        raise ValueError('Either adata or coordinate_df must be provided.')
    
    if coordinate_df is None:
        coordinate_df = adata.uns[coordinates_key]

    if roi_df is None:
        roi_df = adata.uns['rois']

    if genes is None:
        genes = sorted(coordinate_df[gene_key].unique())

    if signature_matrix is None:
        signature_matrix = pd.DataFrame(index=genes,columns=genes).astype(float)
        signature_matrix[:] = np.eye(len(genes))

    if type(n_cases) is int:
        n_cases = list(range(0,n_cases))

    rois_celltyping_x,rois_celltyping_y = get_rois(coordinate_df, genes = genes, min_distance=celltyping_min_expression,
                           KDE_bandwidth=KDE_bandwidth, min_expression=celltyping_min_distance)


    localmax_celltyping_samples =  get_expression_vectors_at_rois(coordinate_df,rois_celltyping_x,rois_celltyping_y,genes,) 

    localmax_celltyping_samples = localmax_celltyping_samples/(localmax_celltyping_samples.to_numpy()**2).sum(0,keepdims=True)**0.5


    dr = dim_reduction(n_components=100)
    factors = dr.fit_transform(localmax_celltyping_samples.T)

    embedder_2d = umap.UMAP(n_components=2,min_dist=0.0)
    embedding = embedder_2d.fit_transform(factors)

    embedder_3d = umap.UMAP(n_components=3, min_dist=0.0,n_neighbors=10,
                    init=np.concatenate([embedding,0.1*np.random.normal(size=(embedding.shape[0],1))],axis=1))
    embedding_color = embedder_3d.fit_transform(embedding)

    embedding_color,color_pca = fill_color_axes(embedding_color)

    color_min = embedding_color.min(0)
    color_max = embedding_color.max(0)

    colors = min_to_max(embedding_color.copy())

    # plt.figure(figsize=(5,5))
    
    def determine_celltype_class_assignments(expression_samples):
        correlations = np.array([np.corrcoef(expression_samples.iloc[:,i],signature_matrix.values.T)[0,1:] for i in range(expression_samples.shape[1])])
        return np.argmax(correlations,-1)

    celltypes = sorted(signature_matrix.columns)
    celltype_class_assignments = determine_celltype_class_assignments(localmax_celltyping_samples)
    print(celltype_class_assignments)
    # determine the center of gravity of each celltype in the embedding:
    celltype_centers = np.array([np.median(embedding[celltype_class_assignments==i,:],axis=0) for i in range(len(celltypes))])

    divergence_indices = np.argsort(roi_df.divergence.values)[::-1]

    for n_case in n_cases:
        x,y = (roi_df.x[divergence_indices[n_case]],roi_df.y[divergence_indices[n_case]])

        # ct_top,ct_bottom = get_celltype(expressions_top.iloc[idcs[n_case]]),get_celltype(expressions_bottom.iloc[idcs[n_case]])

        print("Plotting case {}".format(n_case))

        subsample_mask = get_spatial_subsample_mask(coordinate_df,x,y,plot_window_size=plot_window_size)
        subsample = coordinate_df[subsample_mask]

        distances, neighbor_indices = create_knn_graph(subsample[['x','y','z']].values,k=90)
        local_expression = get_knn_expression(distances,neighbor_indices,genes,subsample.gene.cat.codes.values,bandwidth=1.0)
        local_expression = local_expression/((local_expression**2).sum(0)**0.5)
        subsample_embedding, subsample_embedding_color = transform_embeddings(local_expression.T.values,dr,embedder_2d=embedder_2d,embedder_3d=embedder_3d)
        subsample_embedding_color,_ = fill_color_axes(subsample_embedding_color,color_pca)
        subsample_embedding_color = (subsample_embedding_color-color_min)/(color_max-color_min)
        subsample_embedding_color = np.clip(subsample_embedding_color,0,1)

        plt.figure(figsize=(18,12))

        # plt.suptitle('-'.join([str(ct_top),str(ct_bottom)]))

        ax1 = plt.subplot(234,projection='3d')
        ax1.scatter(subsample.x,subsample.y,subsample.z,c=subsample_embedding_color,marker='.',alpha=0.1)
        ax1.set_zlim(np.median(subsample.z)-plot_window_size,np.median(subsample.z)+plot_window_size)

        ax2 = plt.subplot(231)
        plt.scatter(embedding[:,0],embedding[:,1],c='lightgrey',alpha=0.05,marker='.')
        plot_embeddings(subsample_embedding,subsample_embedding_color,celltype_centers,celltypes)
        
        ax3 = plt.subplot(235)
        # plt.imshow((divergence*hist_sum).T,cmap='Greys', alpha=0.3 )
        ax3.scatter(subsample[subsample.z>subsample.z_delim].x,subsample[subsample.z>subsample.z_delim].y,
        c=subsample_embedding_color[subsample.z>subsample.z_delim],marker='.',alpha=0.1,s=20)
        ax3.set_xlim(x-plot_window_size,x+plot_window_size)
        ax3.set_ylim(y-plot_window_size,y+plot_window_size)

        ax3 = plt.subplot(236)    
        # plt.imshow(hist_sum.T,cmap='Greys',alpha=0.3 )
        ax3.scatter(subsample[subsample.z<subsample.z_delim].x,subsample[subsample.z<subsample.z_delim].y,
        c=subsample_embedding_color[subsample.z<subsample.z_delim],marker='.',alpha=0.1,s=20)
        ax3.set_xlim(x-plot_window_size,x+plot_window_size)
        ax3.set_ylim(y-plot_window_size,y+plot_window_size)
        ax4 = plt.subplot(232)
        plt.scatter(coordinate_df.x,coordinate_df.y,c='k',alpha=0.01,marker='.',s=1)
        plt.scatter(subsample.x,subsample.y,c=subsample_embedding_color,marker='.',alpha=0.8,s=1)

