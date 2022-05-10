import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import stats

from lib.import_funcs import *


def scatter_plot_allGE(matx, gene_a, gene_b, path):
    """This function plot the scatter plot of the GE measured over different cells for two specific genes.
    - matx is the matrix at all times, it is diveded in submatrices using "initial_idx.csv" and "final_idx.csv", 
      that give the indices to split the matrix
    - gene_a and gene_b are the selected genes
    - path is the main folder
    --> it returns the plot of all the time steps, hilighted by different colors
        + the correlation coefficient computed on the whole matrix"""
    
    genes = load_genes(path)
    initial_idx = np.loadtxt(path+"general_info/cells_times/initial_idx.csv", dtype="int")
    final_idx = np.loadtxt(path+"general_info/cells_times/final_idx.csv", dtype="int")
    
    a = [[matx[:,initial_idx[ii]:final_idx[ii]+1]] for ii in range(len(initial_idx))]
    
    # all together
    plt.figure(figsize=(12,9))
    plt.scatter(np.squeeze(np.squeeze(np.array(a[0]))[np.where(genes == gene_a)[0]]), np.squeeze(np.squeeze(np.array(a[0]))[np.where(genes == gene_b)[0]]))
    plt.scatter(np.squeeze(np.squeeze(np.array(a[1]))[np.where(genes == gene_a)[0]]), np.squeeze(np.squeeze(np.array(a[1]))[np.where(genes == gene_b)[0]]))
    plt.scatter(np.squeeze(np.squeeze(np.array(a[2]))[np.where(genes == gene_a)[0]]), np.squeeze(np.squeeze(np.array(a[2]))[np.where(genes == gene_b)[0]]))
    plt.scatter(np.squeeze(np.squeeze(np.array(a[3]))[np.where(genes == gene_a)[0]]), np.squeeze(np.squeeze(np.array(a[3]))[np.where(genes == gene_b)[0]]))
    plt.scatter(np.squeeze(np.squeeze(np.array(a[4]))[np.where(genes == gene_a)[0]]), np.squeeze(np.squeeze(np.array(a[4]))[np.where(genes == gene_b)[0]]))
    plt.grid()
    plt.xlabel(str(gene_a))
    plt.ylabel(str(gene_b))
    
    plt.title("all" + ", " + str(gene_a)+ "->"+ str(gene_b))
    plt.show()
    Sp_corr_act = sp.stats.spearmanr(np.squeeze(matx[np.where(genes == gene_a)[0]]), np.squeeze(matx[np.where(genes == gene_b)[0]]))[0]
    return(np.round(Sp_corr_act,4))


def scatter_plot_GE(matx, gene_a, gene_b, path, text, all_times = True):
    """This function plot the scatter plot of the GE measured over different cells for two specific genes.
    - matx is the matrix at all times, it is diveded in submatrices using "initial_idx.csv" and "final_idx.csv", 
      that give the indices to split the matrix
    - gene_a and gene_b are the selected genes
    - path is the main folder
    --> it returns the plots of all the time steps,  and the plots of all the time steps together hilighted by different colors
        + the correlation coefficient computed on the whole matrix"""
    
    genes = load_genes(path)
    initial_idx = np.loadtxt(path+"general_info/cells_times/initial_idx.csv", dtype="int")
    final_idx = np.loadtxt(path+"general_info/cells_times/final_idx.csv", dtype="int")
    
    a = [[matx[:,initial_idx[ii]:final_idx[ii]+1]] for ii in range(len(initial_idx))]
    
    fig, axs = plt.subplots(3, 2, figsize=(15,15))
    
    axs[0, 0].scatter(np.squeeze(np.squeeze(np.array(a[0]))[np.where(genes == gene_a)[0]]), np.squeeze(np.squeeze(np.array(a[0]))[np.where(genes == gene_b)[0]]))
    axs[0, 0].grid()
    # axs[0, 0].set_xlim([-0.2,3.5])
    # axs[0, 0].set_ylim([-0.2,4])
    axs[0, 0].set_xlabel(str(gene_a))
    axs[0, 0].set_ylabel(str(gene_b))
    axs[0, 0].set_title("00h")
    
    axs[0, 1].scatter(np.squeeze(np.squeeze(np.array(a[1]))[np.where(genes == gene_a)[0]]), np.squeeze(np.squeeze(np.array(a[1]))[np.where(genes == gene_b)[0]]))
    axs[0, 1].grid()
    # axs[0, 1].set_xlim([-0.2,3.5])
    # axs[0, 1].set_ylim([-0.2,4])
    axs[0, 1].set_xlabel(str(gene_a))
    axs[0, 1].set_ylabel(str(gene_b))
    axs[0, 1].set_title("06h")
    
    axs[1, 0].scatter(np.squeeze(np.squeeze(np.array(a[2]))[np.where(genes == gene_a)[0]]), np.squeeze(np.squeeze(np.array(a[2]))[np.where(genes == gene_b)[0]]))
    axs[1, 0].grid()
    # axs[1, 0].set_xlim([-0.2,3.5])
    # axs[1, 0].set_ylim([-0.2,4])
    axs[1, 0].set_xlabel(str(gene_a))
    axs[1, 0].set_ylabel(str(gene_b))
    axs[1, 0].set_title("12h")
    
    axs[1, 1].scatter(np.squeeze(np.squeeze(np.array(a[3]))[np.where(genes == gene_a)[0]]), np.squeeze(np.squeeze(np.array(a[3]))[np.where(genes == gene_b)[0]]))
    axs[1, 1].grid()
    # axs[1, 1].set_xlim([-0.2,3.5])
    # axs[1, 1].set_ylim([-0.2,4])
    axs[1, 1].set_xlabel(str(gene_a))
    axs[1, 1].set_ylabel(str(gene_b))
    axs[1, 1].set_title("24h")
    
    axs[2, 0].scatter(np.squeeze(np.squeeze(np.array(a[4]))[np.where(genes == gene_a)[0]]), np.squeeze(np.squeeze(np.array(a[4]))[np.where(genes == gene_b)[0]]))
    axs[2, 0].grid()
    # axs[2, 0].set_xlim([-0.2,3.5])
    # axs[2, 0].set_ylim([-0.2,4])
    axs[2, 0].set_xlabel(str(gene_a))
    axs[2, 0].set_ylabel(str(gene_b))
    axs[2, 0].set_title("48h")
    
    # all together
    if all_times == True:
        axs[2, 1].scatter(np.squeeze(np.squeeze(np.array(a[0]))[np.where(genes == gene_a)[0]]), np.squeeze(np.squeeze(np.array(a[0]))[np.where(genes == gene_b)[0]]))
        axs[2, 1].scatter(np.squeeze(np.squeeze(np.array(a[1]))[np.where(genes == gene_a)[0]]), np.squeeze(np.squeeze(np.array(a[1]))[np.where(genes == gene_b)[0]]))
        axs[2, 1].scatter(np.squeeze(np.squeeze(np.array(a[2]))[np.where(genes == gene_a)[0]]), np.squeeze(np.squeeze(np.array(a[2]))[np.where(genes == gene_b)[0]]))
        axs[2, 1].scatter(np.squeeze(np.squeeze(np.array(a[3]))[np.where(genes == gene_a)[0]]), np.squeeze(np.squeeze(np.array(a[3]))[np.where(genes == gene_b)[0]]))
        axs[2, 1].scatter(np.squeeze(np.squeeze(np.array(a[4]))[np.where(genes == gene_a)[0]]), np.squeeze(np.squeeze(np.array(a[4]))[np.where(genes == gene_b)[0]]))
        axs[2, 1].grid()
        #axs[2, 1].set_xlim([-0.2,3.5])
        #axs[2, 1].set_ylim([-0.2,4])
        axs[2, 1].set_xlabel(str(gene_a))
        axs[2, 1].set_ylabel(str(gene_b))
        axs[2, 1].set_title("all")
    
    fig.suptitle(text + ", " + str(gene_a)+ "->"+ str(gene_b))
        
    Sp_corr_act = sp.stats.spearmanr(np.squeeze(matx[np.where(genes == gene_a)[0]]), np.squeeze(matx[np.where(genes == gene_b)[0]]))[0]
    return(np.round(Sp_corr_act,4))