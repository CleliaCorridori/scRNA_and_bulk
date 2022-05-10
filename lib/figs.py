import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import stats
from lib.import_funcs import *


def hist_GEmatx(matx, xlab, title, pdf = True, lin = False, together = False, leg="", alpha_c = 0.9, zeros = True, thr = 10**(-5)):
    if lin == False:
        lin_matx = np.reshape(matx, (np.shape(matx)[0]* np.shape(matx)[1]))
    if lin == True:
        lin_matx = np.copy(matx)
    
    if zeros == False:
        lin_matx[abs(lin_matx)<=thr]=float("NAN")
        #print( np.sum(np.isnan(lin_matx)))
    #print("min: ", np.nanmin(lin_matx), "\nmax: ", np.nanmax(lin_matx))
    
    if together == False:
        plt.figure(figsize=(9,5))
    n, bins, patches = plt.hist(lin_matx, 100, density= pdf, alpha=alpha_c, label=leg)
    plt.xlabel(xlab)
    if pdf == True:
        plt.ylabel('pdf')
    else:
        plt.ylabel('counts')
    if len(leg)>2:
        plt.legend()   
    plt.title(title)
    plt.grid()
    if together == False:
        plt.show()
        
def zeros_hist(matx, cell=1, pdf = False, thr=10**(-3), alpha_p = 1):
    """cell == 1 --> number of zeros per cell (numero di geni spenti per cellula)
       cell == 0 --> number of zeros per gene (numero di cellule in cui non misuro il gene)"""
    # setting to NaN the zeros values to count them
    matx[abs(matx)<=thr]=float("NAN")
    if  cell == 1:
        Nzeros = [np.sum(np.isnan(matx[:,ii])) for ii in range(np.shape(matx)[cell])]
    else: 
        Nzeros = [np.sum(np.isnan(matx[ii,:])) for ii in range(np.shape(matx)[cell])]
        
    # select the number of bins: Freedman-Diaconis rule
    distr = pd.DataFrame(np.squeeze(Nzeros))
    q1 = distr.quantile(0.25)
    q3 = distr.quantile(0.75)
    iqr = q3 - q1
    len_distr = len(distr[~np.isnan(distr)])
    bin_width = (2 * iqr) / (len_distr ** (1 / 3))
    bin_count = int(np.ceil((np.nanmax(distr) - np.nanmin(distr)) / bin_width))
            # bin_count = int(np.ceil(np.log2(len_distr)) + 1) #second method to compute Nbins

    plt.figure(figsize=(10,7))
    n, b, c = plt.hist(Nzeros, bin_count, density=pdf, alpha=alpha_p)   

    if cell == 1:
        plt.xlabel('# of gene=0 per cell')
        plt.title("Number of 0s per cell")
    else:
        plt.xlabel("# of cell=0 per gene")
        plt.title("Number of 0s per gene")
    if pdf == True:
        plt.ylabel('pdf')
    else:
        plt.ylabel('counts')
    plt.show()
        
#-------------------------------------
#        VIOLIN PLOTS
#-------------------------------------
def violin_GE(matx, path, violin=True):
    # initial info
    times = ["00", "06", "12", "24", "48", "all"]
    genes = load_genes(path) # gene list
    a = split_time_matrices(matx, path) # Gene expression matrices divided by time steps
    imp_genes = np.loadtxt(path+"general_info/imp_genes.csv", dtype="str")

    for kk in range(len(imp_genes)):
        gene_d = imp_genes[kk]
        idx = np.where (genes == gene_d)[0]

        # plotting violin plot for the selected gene:
        data_to_plot = [np.squeeze(a[0][idx,:]), 
                        np.squeeze(a[1][idx,:]), 
                        np.squeeze(a[2][idx,:]), 
                        np.squeeze(a[3][idx,:]), 
                        np.squeeze(a[4][idx,:]), 
                        np.squeeze(matx[idx, :])]

        # Create a figure instance
        fig = plt.figure(figsize=(25,8))

        # Create the boxplot
        if violin==True:
            plt.violinplot(data_to_plot, showmeans=True, showmedians=True)
        elif violin==False:
            plt.boxplot(data_to_plot,  showmeans=True)
        else: 
            plt.boxplot(data_to_plot)#, showmeans=True)
            plt.violinplot(data_to_plot, showmedians=True)
           
        plt.xticks(ticks=np.linspace(1, len(times), len(times)), labels=times, rotation = 90, fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlabel("time [h]", fontsize=20)
        plt.ylabel("Normalized GE", fontsize=20)
        plt.title("Gene "+str(gene_d), fontsize=22)
        plt.savefig(path+'out_img/N_GE_violinplot/N_GE_'+str(gene_d)+'.png', bbox_inches='tight')
        
def superimposed_violin_GE(matx1, matx2, path):
    # initial info
    times = ["00", "06", "12", "24", "48", "all"]
    genes = load_genes(path) # gene list
    a1 = split_time_matrices(matx1, path) # Gene expression matrices divided by time steps
    a2 = split_time_matrices(matx2, path) # Gene expression matrices divided by time steps
    imp_genes = np.loadtxt(path+"general_info/imp_genes.csv", dtype="str")

    for kk in range(len(imp_genes)):
        gene_d = imp_genes[kk]
        idx = np.where (genes == gene_d)[0]

        # plotting violin plot for the selected gene:
        data_to_plot1 = [np.squeeze(a1[0][idx,:]), 
                         np.squeeze(a1[1][idx,:]), 
                         np.squeeze(a1[2][idx,:]), 
                         np.squeeze(a1[3][idx,:]), 
                         np.squeeze(a1[4][idx,:]), 
                         np.squeeze(matx1[idx, :])]
        data_to_plot2 = [np.squeeze(a2[0][idx,:]), 
                         np.squeeze(a2[1][idx,:]), 
                         np.squeeze(a2[2][idx,:]), 
                         np.squeeze(a2[3][idx,:]), 
                         np.squeeze(a2[4][idx,:]), 
                         np.squeeze(matx2[idx, :])]

        # Create a figure instance
        fig = plt.figure(figsize=(25,8))

        # Create the boxplot
        plt.violinplot(data_to_plot1, showmeans=True)
        plt.violinplot(data_to_plot2, showmeans=True)
        
        plt.xticks(ticks=np.linspace(1, len(times), len(times)), labels=times, rotation = 90, fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlabel("time [h]", fontsize=20)
        plt.ylabel("Normalized GE", fontsize=20)
        plt.title("Gene "+str(gene_d), fontsize=22)
        plt.savefig(path+'out_img/N_GE_violinplot/N_GE_'+str(gene_d)+'.png', bbox_inches='tight')
        

#-------------------------------------
#        Distributions
#-------------------------------------
def distr_GE_perGene(matx_orig, imp_genes, path, title, pdf = True, zeros = True, thr = 10**(-3), in_t=0, fin_t=5):
    fin_t += 1
    print(title)
    matx = np.copy(matx_orig)
    if zeros == False:
        # setting to NaN the zeros values
        matx[abs(matx)<=thr]=float("NAN")
        lin_matx = np.ndarray.flatten(matx)
        print("Number of values below the threshold in all data(", thr, "):", len(lin_matx[np.isnan(lin_matx)]), 
              "\nfraction of zeros: ", len(lin_matx[np.isnan(lin_matx)])/len(lin_matx)) # ~ to use the negation
        
        
    # initial info
    times = ["00", "06", "12", "24", "48", "all"]
    genes = load_genes(path) # gene list
    a = split_time_matrices(matx, path) # Gene expression matrices divided by time steps  
    
    
    if len(imp_genes) == 1:
        gene_d = imp_genes[0]
        idx = np.where (genes == gene_d)[0]
        # select the number of bins: Freedman-Diaconis rule
        distr = pd.DataFrame(np.squeeze(matx[idx, :]))
        q1 = distr.quantile(0.25)
        q3 = distr.quantile(0.75)
        iqr = q3 - q1
        len_distr = len(distr[~np.isnan(distr)])
        bin_width = (2 * iqr) / (len_distr ** (1 / 3))
        bin_count = int(np.ceil((np.nanmax(distr) - np.nanmin(distr)) / bin_width))
        # bin_count = int(np.ceil(np.log2(len_distr)) + 1) #second method to compute Nbins

        fig, axs = plt.subplots(fin_t+1-in_t,1, figsize=(10,3*(fin_t+1-in_t)))
        [axs[ii].hist(np.squeeze(a[ii][idx,:]), bin_count, [np.nanmin(distr), np.nanmax(distr)], 
                      density= pdf, alpha=0.7) for ii in range(fin_t-in_t)]
        
        axs[fin_t-in_t].hist(np.squeeze(matx[idx, :]), bin_count, [np.nanmin(distr), np.nanmax(distr)],
                             density= pdf, alpha=0.7)
        axs[fin_t-in_t].set_title(str(gene_d)+", all times ")
        [axs[ii-in_t].set_title(str(gene_d)+", time "+ str(times[ii])) for ii in range(in_t, fin_t)]
        [axs[ii].set_xlabel("Gene_expression") for ii in range(fin_t-in_t)]
        if pdf == True:
            [axs[ii].set_ylabel('pdf') for ii in range(fin_t-in_t)]
        else:
            [axs[ii].set_ylabel('counts') for ii in range(fin_t-in_t)]


        #fig.suptitle(title)
        fig.tight_layout()
        fig.subplots_adjust(top=0.88)
        plt.show()
        
    else: 
        for kk in range(len(imp_genes)):
            gene_d = imp_genes[kk]
            idx = np.where (genes == gene_d)[0]
            # select the number of bins: Freedman-Diaconis rule
            distr = pd.DataFrame(np.squeeze(matx[idx, :]))
            q1 = distr.quantile(0.25)
            q3 = distr.quantile(0.75)
            iqr = q3 - q1
            len_distr = len(distr[~np.isnan(distr)])
            bin_width = (2 * iqr) / (len_distr ** (1 / 3))
            bin_count = int(np.ceil((np.nanmax(distr) - np.nanmin(distr)) / bin_width))
            # bin_count = int(np.ceil(np.log2(len_distr)) + 1) #second method to compute Nbins

            # Create a figure instance
            fig, axs = plt.subplots(fin_t+1-in_t,1, figsize=(10,3*(fin_t+1-in_t)))

            # Create the boxplot
            [axs[ii].hist(np.squeeze(a[ii][idx,:]), bin_count,  [np.nanmin(distr), np.nanmax(distr)], 
                          density= pdf, alpha=0.7) for ii in range(fin_t-in_t)]
            
            axs[fin_t-in_t].hist(np.squeeze(matx[idx, :]), bin_count,  [np.nanmin(distr), np.nanmax(distr)],  
                                 density= pdf, alpha=0.7)
            axs[fin_t-in_t].set_title(str(gene_d)+", all times ")
            [axs[ii-in_t].set_title(str(gene_d)+", time "+ str(times[ii])) for ii in range(in_t, fin_t)]
            [axs[ii].set_xlabel("Gene_expression") for ii in range(fin_t-in_t)]
            if pdf == True:
                [axs[ii].set_ylabel('pdf') for ii in range(fin_t-in_t)]
            else:
                [axs[ii].set_ylabel('counts') for ii in range(fin_t-in_t)]


            #fig.suptitle(title)
            fig.tight_layout()
            fig.subplots_adjust(top=0.88)
            plt.show()



def distr_GE_perGene_comparison(matx_orig1, matx_orig2, imp_genes, path, title1, title2, pdf = True, zeros = True, thr = 10**(-3), in_t=0, fin_t=5):
    fin_t += 1
    print(title1 + " and " + title2)
    matx1 = np.copy(matx_orig1)
    matx2 = np.copy(matx_orig2)
    if zeros == False:
        # setting to NaN the zeros values
        matx1[abs(matx1)<=thr]=float("NAN")
        lin_matx1 = np.ndarray.flatten(matx1)
        
        matx2[abs(matx2)<=thr]=float("NAN")
        lin_matx2 = np.ndarray.flatten(matx2)
        
        print("Number of values below the threshold in all data (", thr, "):", len(lin_matx1[np.isnan(lin_matx1)]), 
              "\nfraction of zeros: ", len(lin_matx1[np.isnan(lin_matx1)])/len(lin_matx1))
        print("Number of values below the threshold in all data (", thr, "):", len(lin_matx2[np.isnan(lin_matx2)]), 
              "\nfraction of zeros: ", len(lin_matx2[np.isnan(lin_matx2)])/len(lin_matx2))
        
        
    # initial info
    times = ["00", "06", "12", "24", "48", "all"]
    genes = load_genes(path) # gene list
    a1 = split_time_matrices(matx1, path) # Gene expression matrices divided by time steps  
    a2 = split_time_matrices(matx2, path) # Gene expression matrices divided by time steps  
    
    
    if len(imp_genes) == 1:
        gene_d = imp_genes[0]
        idx = np.where (genes == gene_d)[0]
        # select the number of bins: Freedman-Diaconis rule-----------------
        distr1 = pd.DataFrame(np.squeeze(matx1[idx, :]))
        distr2 = pd.DataFrame(np.squeeze(matx2[idx, :]))
        
        if (np.nanmax(distr1)-np.nanmin(distr1))<=(np.nanmax(distr2)-np.nanmin(distr2)):
            distr = pd.DataFrame(np.copy(distr2))
        else:
            distr = pd.DataFrame(np.copy(distr1))
            
        q1 = distr.quantile(0.25) #1
        q3 = distr.quantile(0.75)
        iqr = q3 - q1        
        len_distr = len(distr[~np.isnan(distr)])
        bin_width = (2 * iqr) / (len_distr ** (1 / 3))
        bin_count = int(np.ceil((np.nanmax(distr) - np.nanmin(distr)) / bin_width))
        # ----------------------------------------------------------------------
        
        fig, axs = plt.subplots(fin_t+1-in_t,1, figsize=(10,3*(fin_t+1-in_t)))
        [axs[ii].hist(np.squeeze(a1[ii][idx,:]), bin_count, [np.nanmin(distr), np.nanmax(distr)], 
                      density= pdf, alpha=0.7, label=title1) for ii in range(fin_t-in_t)]
        [axs[ii].hist(np.squeeze(a2[ii][idx,:]), bin_count, [np.nanmin(distr), np.nanmax(distr)], 
                      density= pdf, alpha=0.7, label=title2) for ii in range(fin_t-in_t)]
        [axs[ii].legend(loc='upper right') for ii in range(fin_t-in_t)]
        
        axs[fin_t-in_t].hist(np.squeeze(matx1[idx, :]), bin_count, [np.nanmin(distr), np.nanmax(distr)],
                             density= pdf, alpha=0.7, label=title1)
        axs[fin_t-in_t].hist(np.squeeze(matx2[idx, :]), bin_count, [np.nanmin(distr), np.nanmax(distr)],
                             density= pdf, alpha=0.7, label=title2)
        axs[fin_t-in_t].legend(loc='upper right')
        axs[fin_t-in_t].set_title(str(gene_d)+", all times ")
        [axs[ii-in_t].set_title(str(gene_d)+", time "+ str(times[ii])) for ii in range(in_t, fin_t)]
        [axs[ii].set_xlabel("Gene_expression") for ii in range(fin_t-in_t)]
        if pdf == True:
            [axs[ii].set_ylabel('pdf') for ii in range(fin_t-in_t)]
        else:
            [axs[ii].set_ylabel('counts') for ii in range(fin_t-in_t)]


        #fig.suptitle(title)
        fig.tight_layout()
        fig.subplots_adjust(top=0.88)
        plt.show()
        
    else: 
        for kk in range(len(imp_genes)):
            gene_d = imp_genes[kk]
            idx = np.where (genes == gene_d)[0]
            # select the number of bins: Freedman-Diaconis rule-----------------
            distr1 = pd.DataFrame(np.squeeze(matx1[idx, :]))
            distr2 = pd.DataFrame(np.squeeze(matx2[idx, :]))

            if (np.nanmax(distr1)-np.nanmin(distr1))<=(np.nanmax(distr2)-np.nanmin(distr2)):
                distr = pd.DataFrame(np.copy(distr2))
            else:
                distr = pd.DataFrame(np.copy(distr1))

            q1 = distr.quantile(0.25) #1
            q3 = distr.quantile(0.75)
            iqr = q3 - q1
            len_distr = len(distr[~np.isnan(distr)])
            bin_width = (2 * iqr) / (len_distr ** (1 / 3))
            bin_count = int(np.ceil((np.nanmax(distr) - np.nanmin(distr)) / bin_width))
            # ----------------------------------------------------------------------

            # Create a figure instance
            fig, axs = plt.subplots(fin_t+1-in_t,1, figsize=(10,3*(fin_t+1-in_t)))
            [axs[ii].hist(np.squeeze(a1[ii][idx,:]), bin_count, [np.nanmin(distr), np.nanmax(distr)], 
                          density= pdf, alpha=0.7, label=title1) for ii in range(fin_t-in_t)]
            [axs[ii].hist(np.squeeze(a2[ii][idx,:]), bin_count, [np.nanmin(distr), np.nanmax(distr)], 
                          density= pdf, alpha=0.7, label=title2) for ii in range(fin_t-in_t)]
            [axs[ii].legend(loc='upper right') for ii in range(fin_t-in_t)]

            axs[fin_t-in_t].hist(np.squeeze(matx1[idx, :]), bin_count, [np.nanmin(distr), np.nanmax(distr)],
                                 density= pdf, alpha=0.7, label=title1)
            axs[fin_t-in_t].hist(np.squeeze(matx2[idx, :]), bin_count, [np.nanmin(distr), np.nanmax(distr)],
                                 density= pdf, alpha=0.7, label=title2)
            axs[fin_t-in_t].legend(loc='upper right')
            axs[fin_t-in_t].set_title(str(gene_d)+", all times ")
            [axs[ii-in_t].set_title(str(gene_d)+", time "+ str(times[ii])) for ii in range(in_t, fin_t)]
            [axs[ii].set_xlabel("Gene_expression") for ii in range(fin_t-in_t)]
            if pdf == True:
                [axs[ii].set_ylabel('pdf') for ii in range(fin_t-in_t)]
            else:
                [axs[ii].set_ylabel('counts') for ii in range(fin_t-in_t)]


            #fig.suptitle(title)
            fig.tight_layout()
            fig.subplots_adjust(top=0.88)
            plt.show()
 


 #----------
def plot_GE(matx, ax_names, text, n_cells=50, n_genes=24):
    """ function to plot GE matrix with the selected Ngenes and Ncells, hilighting the presence of zeros"""
    y_pos = np.arange(0,np.shape(matx[:n_genes])[0])
    ax_names = ax_names[:n_genes]
    submatrix = matx[:n_genes, :n_cells]
    plt.figure(figsize=(13,6))
    plt.imshow(np.log(submatrix+1))
    plt.title(text)
    plt.xlabel("cells")
    plt.ylabel("genes")
    #plt.xticks(x_pos, labels=ax_names, rotation='vertical', fontsize=18)
    plt.yticks(y_pos, labels=ax_names) #, fontsize=18)
    plt.colorbar()
    plt.show()
    
def fig_matx(matx, ax_names, titl):
    """function to plot the correlation Matrix (Pearson) with gene names"""
    x_pos = np.arange(0,np.shape(matx)[0])
    
    plt.figure(figsize=(12,10))
    plt.imshow(matx)
    plt.xticks(x_pos, labels=ax_names, rotation='vertical', fontsize=18)
    plt.yticks(x_pos, ax_names, fontsize=18)
    plt.title(titl, fontsize=18)
    plt.colorbar()
       
#------------
def baloon_plot(matx, path, text, Norm_z=True, cost_z=2000, Norm_GE=False, thr = 10**(-5)):
    matx_splitted = GE_time_matrices(matx, path,  text)
    imp_genes = np.loadtxt(path+"general_info/imp_genes.csv", dtype="str")

    # Computing zeros
    zeros = []
    for ii in range(5):
        for jj in range(np.shape(matx_splitted[ii])[0]):
            zeros.append(np.shape(np.where(matx_splitted[ii][jj, :]<=thr))[1])
    zeros = np.array(zeros, dtype=float) #number of cells=0 per gene
    
    #computing the average GE per gene (averaging over cells)
    matx_Cmean = np.squeeze([[np.mean(matx_splitted[ii], axis=1)] for ii in range(5)])
    if Norm_GE==True:
        matx_Cmean = matx_Cmean/(np.max(matx_Cmean))
    
    #reshaping for the plot
    time_idx = np.reshape([np.ones(np.shape(matx_Cmean)[1])*np.arange(np.shape(matx_Cmean)[0])[ii] for ii in range(np.shape(matx_Cmean)[0])], (np.shape(matx_Cmean)[1]*np.shape(matx_Cmean)[0]))
    gene_idx = np.tile(np.arange(0,np.shape(matx_Cmean)[1]),np.shape(matx_Cmean)[0])
    matx_Cmean_lin = np.reshape(matx_Cmean, (np.shape(matx_Cmean)[0]*np.shape(matx_Cmean)[1])) #GE, described by the color

    # plot
    if Norm_z==True:
        zeros= zeros/np.max(zeros)*cost_z
    plt.figure(figsize=(40,10))
    plt.scatter(gene_idx, time_idx, c=matx_Cmean_lin, s= zeros, cmap="YlOrRd")
    cbar =plt.colorbar()
    cbar.ax.tick_params(labelsize=22) 
    cbar.set_label('Average gene expression', rotation=270, fontsize=27, verticalalignment='baseline')
    x_pos = np.arange(0,np.shape(imp_genes)[0])
    y_pos = np.arange(0,5)
    plt.xticks(x_pos, labels=imp_genes, rotation='vertical', fontsize=22)
    plt.yticks(y_pos, labels=["00", "06", "12", "24", "48"], fontsize=22)
    plt.xlabel("Genes", fontsize=27)
    plt.ylabel("Time [h]", fontsize=27)
    plt.title(text, fontsize=27)
    plt.gca().invert_yaxis()
    plt.grid( which='major', color='gray', linestyle='--', alpha=0.5)
    plt.show()
    return(matx_Cmean, zeros)
