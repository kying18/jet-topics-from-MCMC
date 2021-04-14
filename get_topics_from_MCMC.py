#!/usr/bin/env python3

# kylie edits:
# ./get_topics_from_MCMC.py "150_1_histograms_pp150_1_pp150_1_zjet" -1 100 8000 7000 1000
# ./get_topics_from_MCMC.py "150_1_histograms_pp150_1_pbpb150_0_10_1" -1 100 8000 7000 1000
# ./get_topics_from_MCMC.py "150_1_histograms_pbpb150_0_10_1_pbpb150_0_10_1_wide" -1 100 8000 7000 1000

# ./get_topics_from_MCMC.py "150_1_histograms_pt100_dijets_x100_pbpb150_0_10_pbpb150_0_10_wide" -1 100 8000 7000 1000
# ./get_topics_from_MCMC.py "150_1_histograms_pt100_dijets_x100_pp150_pbpb150_0_10" -1 100 8000 7000 1000

#############################################################################################
#############################################################################################
# Developed by Jasmine Brewer and Andrew Turner with conceptual oversight from Jesse Thaler #
# Last modified 08-18-2020 ##################################################################
#############################################################################################
#############################################################################################

#### A valuable tutorial on fitting with MCMC: https://emcee.readthedocs.io/en/stable/tutorials/line/ ####

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import scipy.optimize as op
from scipy import special
from scipy.stats import wasserstein_distance
from scipy.spatial import distance
import random
import emcee
import math
import ipdb
import pickle
from pathlib import Path
import argparse
import os
import errno
import csv 

DO_MCMC = False
# MASK_FACTOR = 0
# DECENT_STATS_FACTOR = 0.0001
FOLDER_PREFIX = "all"

######################
## importing data directly from csv
######################

def get_data(input_filename, sample1_label, sample2_label):
    # input_filename is the path to the csv file
    # sample1_label/sample2_label is string that represents that we want to use (ie this is the label on the LHS of the csv, like pbpb150_0_10_wide_zjet)
    # sample1_label should probably be different than sample2_label if you want to run this code on multiple samples lol

    samples = {}

    suffixes = ["", "_error", "_quark", "_quark_error", "_gluon", "_gluon_error"]

    with open(input_filename, 'rt') as f:
        all_lines = f.readlines()
        for i in range(len(all_lines)):
            split_vals = all_lines[i].split(",")
            for suffix in suffixes:
                if split_vals[0] == sample1_label+suffix:
                    samples['sample1'+suffix] = np.array(list(map(lambda x: float(x), split_vals[1:-1])))
                elif split_vals[0] == sample2_label+suffix:
                    samples['sample2'+suffix] = np.array(list(map(lambda x: float(x), split_vals[1:-1])))
    # print(samples)
    return samples  # should be a dictionary {'sample1': [histogram values], 'sample1_error': [histogram of errors], ...}

def format_samples(samples, min_bin, max_bin):
    # takes the dictionary of samples and creates normalized histogram: [[normalized histogram, normalized histogram errors, histogram of counts], total count]
    # min_bin and max_bin are indices of minimum bin and maximum bin: [min_bin, max_bin)
    # returns this list of histograms/count for sample1, sample2, combined quark, combined gluon (in that order)
    def format_hist(hist, hist_error):
        # hist is list of integers rep histogram bins
        hist = hist[min_bin:max_bin]
        hist_error = hist_error[min_bin:max_bin]
        tot_n = sum(hist)
        normalized_hist = 1/tot_n * hist
        normalized_bin_error = 1/tot_n * hist_error

        return [[normalized_hist, normalized_bin_error, hist], tot_n]
    
    sample1 = format_hist(samples['sample1'], samples['sample1_error'])
    sample2 = format_hist(samples['sample2'], samples['sample2_error'])
    quarks = format_hist(samples['sample1_quark'] + samples['sample2_quark'], np.sqrt(np.square(samples['sample1_quark_error']) + np.square(samples['sample2_quark_error'])))
    gluons = format_hist(samples['sample1_gluon'] + samples['sample2_gluon'], np.sqrt(np.square(samples['sample1_gluon_error']) + np.square(samples['sample2_gluon_error'])))

    return sample1, sample2, quarks, gluons

######################
## fitting function for the MCMC
#####################
def model_func(a1, b1, c1, a2, b2, c2, a3, b3, c3, a4, b4, c4, alpha, beta, gamma, x):
    
    return alpha*pdf_skew_gaussian(a1,b1,c1,x) + beta*pdf_skew_gaussian(a2,b2,c2,x) + gamma*pdf_skew_gaussian(a3,b3,c3,x) + (1-alpha-beta-gamma)*pdf_skew_gaussian(a4,b4,c4,x)


def pdf_skew_gaussian(mu, s, c, x):
    
    return np.exp(-((x-mu)**2)/(2*s**2))*special.erfc(-(c*(x-mu))/(np.sqrt(2)*s))/(s*np.sqrt(2*np.pi))


######################
## specifies which part of the list of fit parameters are "parameters" and which are fractions.
## needs to be changed if the fitting function is changed
#####################
def get_params_and_fracs(theta):
    
    params = theta[:-6]
    fracs1 = theta[-6:-3]
    fracs2 = theta[-3:]
    
    return [params, [fracs1, fracs2]]
    
    
def in_bounds(theta, bounds):
    
    [params, [fracs1, fracs2]] = get_params_and_fracs(theta)
    fractions = np.concatenate((fracs1, fracs2, [1-np.sum(fracs1), 1-np.sum(fracs2)]))

    # parameters must have the specified bounds
    params_in_bounds = [min(bounds[i])<=params[i]<=max(bounds[i]) for i in range(len(params))]
    
    # fraction parameters must be between 0 and 1
    fracs_in_bounds = [0<=fractions[i]<=1 for i in range(len(fractions))]

    return np.all(params_in_bounds)&np.all(fracs_in_bounds)


    
#################################################################
#### Least squares fitting for starting point of MCMC ###########
#################################################################

def func_simul_lsq(theta,x1,y1,x2,y2, bnds=[(-float("inf"), float("inf"))]):
    
    [params, [fracs1, fracs2]] = get_params_and_fracs(theta)
    
    if in_bounds(theta, bnds):
        return np.concatenate( (model_func(*params, *fracs1, x1) - y1, model_func(*params, *fracs2, x2) - y2) )
    else:
        return 10**10
    
def get_simul_fits(bins, hist1, hist2, trytimes, bnds, initial_point):
    
    costnow=np.inf
    
    # try a least squares fit many times with slightly varying initial points, and keep the best one
    for i in range(0,trytimes):
 
        new_initial_point = (1+5e-1*np.random.randn( len(initial_point) ))*initial_point
        # print(new_initial_point)
        
        if bnds==None:
            fit = least_squares(func_simul_lsq, new_initial_point, args=(bins, hist1, bins, hist2))
        else:
            fit = least_squares(func_simul_lsq, new_initial_point, args=(bins, hist1, bins, hist2, bnds))
            
        if costnow>fit['cost']:
            # print(fit['cost'])
            # print(fit)
            fitnow = fit
            costnow = fit['cost']
        
    return fitnow



#########################################
# MCMC ##################################
#########################################

def get_MCMC_samples(x1, y1, y1err, x2, y2, y2err, fit, tot_weights, bnds, variation_factor, ndim, nwalkers, nsamples):
    
    pos = []
    while len(pos)<nwalkers:
        trial_pos = fit*(1 + variation_factor*np.random.randn(ndim))
        if in_bounds(trial_pos, bnds):
            pos.append(trial_pos)
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_simul, args=(x1, y1, y1err, x2, y2, y2err, bnds, tot_weights))
    sampler.run_mcmc(pos, nsamples)
    
    return sampler
    
######################
## primary function to do the MCMC and extract kappa values from the posterior
#####################
def do_MCMC_and_get_kappa(datum1, datum2, min_bin, max_bin, filelabel, system, nwalkers=500, nsamples=20000, burn_in=100, variation_factor=1e-2, trytimes=500, nkappa=1000, bounds=[(0,25),(0,15),(0,5),(0,25),(0,15),(0,5),(0,25),(0,15),(0,5)], fit_init_point=[14,8,2,5,9,4,10,5,5,0.5,0.3,0.5,0.3]):
    bins = range(max_bin - min_bin + 1)
    [[hist1, hist1_errs, hist1_n], totweight1] = datum1
    [[hist2, hist2_errs, hist2_n], totweight2] = datum2
    histbins = get_mean(bins)
    
    # do a simultaneous least-squares fit to the histograms. Used as a starting point for the MCMC    
    bounds = [(0,50),(1,15),(-20,20),(0,50),(1,15),(-20,20),(0,50),(1,15),(-20,20),(0,50),(1,15),(-20,20),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1)]
    bounds = [(0, float("inf")) for i in range(len(bounds))]

    fit_init_point=[20,15,2.5,10,
                    0.5,1.5,5,5,
                    2,15,10,3,
                    1,0.1,0.1,1,
                    0.1,0.1]

    fit = get_simul_fits(histbins, hist2, hist1, trytimes, bounds, fit_init_point)
    
    fitnow = put_fits_in_order( fit['x'], histbins, hist1, histbins, hist2) 
    ndim = len(fitnow)
    
    [params,[fracs1,fracs2]] = get_params_and_fracs(fitnow)
    result1 = np.concatenate( (params, fracs1) )
    result2 = np.concatenate( (params, fracs2) )
    
    # plot the least-squares fit compared to the histograms
    # plt.scatter(histbins, hist1, color='blue', s=3)
    # plt.scatter(histbins, hist2, color='red', s=3)
    # plt.errorbar(histbins, hist1, hist1_errs, color='blue', label='histogram 1', ls='none')
    # plt.errorbar(histbins, hist2, hist2_errs, color='red',label='histogram 2', ls='none')
    plt.errorbar(histbins+min_bin, hist1, hist1_errs, color='blue', label='histogram 1')
    plt.errorbar(histbins+min_bin, hist2, hist2_errs, color='red',label='histogram 2')
    plt.plot(histbins+min_bin,[model_func(*result1,x) for x in histbins],'g--',label='fit 1')
    plt.plot(histbins+min_bin,[model_func(*result2,x) for x in histbins],'m--',label='fit 2')
    plt.xlabel('Constituent multiplicity')
    plt.ylabel('Probability')
    plt.legend()
    plt.xlim((min_bin,max_bin))
    plt.title('Input Histograms')
    current_dir = Path.cwd()
    if DO_MCMC:
        plt.savefig(current_dir / 'plots' / f'{system}' / (filelabel+'_least-squares_fit.png'))

    if DO_MCMC:
        # do the MCMC    
        print('Starting MCMC')
        sampler = get_MCMC_samples(histbins, hist1, hist1_errs, histbins, hist2, hist2_errs, fitnow, [totweight1, totweight2], bnds=bounds, variation_factor=variation_factor, ndim=ndim, nwalkers=nwalkers, nsamples=nsamples)
        print('Finished MCMC')
        
        samples = sampler.get_chain()
        del sampler

        with open("samples.pkl", "wb+") as f:
            pickle.dump(samples, f)
    
    else:
        with open("samples.pkl", "rb") as f:
            samples = pickle.load(f)
    
    # plot MCMC samples
    fig, axes = plt.subplots(ndim, figsize=(10,16), sharex=True)
    for i in range(ndim):
        axes[i].plot(range(0,nsamples),samples[:, :, i], "k", alpha=0.1)
        axes[i].axvline(x=burn_in,color='blue')
    fig.suptitle('MCMC Samples', fontsize=20)
    plt.savefig(current_dir / 'plots' / f'{system}' / (filelabel+'_MCMC_samples.png'))
    
    # randomly sample "nkappa" points from the posterior on which to extract kappa
    all_index_tuples = [ (i,j) for i in range(burn_in,len(samples)) for j in range(len(samples[0])) ]
    index_tuples = random.sample( all_index_tuples, nkappa )
    posterior_samples = np.array( [[samples[ tup[0], tup[1], i] for i in range(ndim)] for tup in index_tuples ] )
    # first_dim_sample = random.choices(range(burn_in, len(samples)), k=nkappa)
    # second_dim_sample = random.choices(range(len(samples[0])), k=nkappa)
    # posterior_samples = np.array( [[samples[ first_dim_sample[j], second_dim_sample[j], i] for i in range(ndim)] for j in range(nkappa) ] )
    del samples
    
    # extract kappa from the posterior, only on points of the fit where at least one input histogram is non-zero
    # or_mask = (hist1_n>5)|(hist2_n>5)
    # or_mask = (hist1_n>MASK_FACTOR*totweight1)|(hist2_n>MASK_FACTOR*totweight2) #&(((hist1_n>2)|(hist2_n>2)))
    or_mask = (hist1_n>0)|(hist2_n>0)
    mask_label = 'or'
    [kappas12, kappas21] = get_kappa(posterior_samples, datum1, datum2, histbins, min_bin, or_mask, filelabel+'kappas_'+mask_label+'.png', system)

    del posterior_samples
    
    return [kappas12,kappas21]
 
    
#################################################################
#### Prior, prob, and likelihood functions for MCMC #############
#################################################################

def lnprior_bnds(theta, bounds):
        
    if in_bounds(theta, bounds):
        return 0.0 
    return -np.inf
    
def lnprob_simul(theta, x1, y1, y1err, x2, y2, y2err, bounds, totweights): 
    
    lp = lnprior_bnds(theta, bounds)
    
    if not np.isfinite(lp):
        return -np.inf
        
    return lp + lnlike_simul(theta, x1, y1, y1err, x2, y2, y2err, totweights)
    
def lnlike_simul(theta, x1, y1, y1err, x2, y2, y2err, totweights):
    
    thetanow = put_fits_in_order(theta, x1, y1, x2, y2)
    [params, [fracs1, fracs2]] = get_params_and_fracs( thetanow )
   
    [totweight1, totweight2] = totweights
    
    return lnlike_individ(y1, y1err, model_func(*params, *fracs1, x1), totweight1) + lnlike_individ(y2, y2err, model_func(*params, *fracs2, x2), totweight2)    

def lnlike_individ(y, yerr, model, totweight):

    vec = [ model[i]-y[i]+y[i]*np.log(y[i])-y[i]*np.log(model[i]) if y[i]>0 else model[i] for i in range(len(y)) ]

    return -totweight*np.sum( vec )

   
######################
## function to extract kappa values given a sampling of the posterior, and a mask specifying where kappa can be extracted
#####################      
def get_kappa(all_samples, datum1, datum2, histbins, min_bin, mask, filelabel, system, upsample_factor=10):
    
    [[hist1, hist1_errs, hist1_n], _] = datum1
    [[hist2, hist2_errs, hist2_n], _] = datum2
    
    kappa12 = np.zeros( len(all_samples) )
    kappa21 = np.zeros( len(all_samples) )
    
    mask1_zeros = hist1_n>0 # used for plotting only
    mask2_zeros = hist2_n>0 # used for plotting only
    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(8, 6))
    ax1.plot(histbins[mask2_zeros]+min_bin,hist1[mask2_zeros]/hist2[mask2_zeros],'b--')
    ax2.plot(histbins[mask1_zeros]+min_bin,hist2[mask1_zeros]/hist1[mask1_zeros],'b--',label='data')

    # "upsample" the histogram bins by upsample_factor to determine the bins on which kappa will be evaluated from the model
    model_bins = np.append( np.concatenate( ( [np.linspace(histbins[i],histbins[i+1],upsample_factor, endpoint=False) for i in range(len(histbins)-1)] ) ), histbins[-1] )

    decent_stats_indices = np.where( (hist1_n>10)&(hist2_n>10) )[0]
    # minimum (left) and maximum (right) indices in model_bins where both histograms have more than 10 data points
    left_decent_stats_cutoff = upsample_factor*np.min( decent_stats_indices )
    right_decent_stats_cutoff = upsample_factor*np.max( decent_stats_indices )

    mask_indices = np.where( mask )
    # minimum (left) and maximum (right) indices in model_bins where the mask is true
    left_mask_cutoff = upsample_factor*np.min( mask_indices )
    right_mask_cutoff = upsample_factor*np.max( mask_indices )
    
    right_bins = model_bins[ left_decent_stats_cutoff:(right_mask_cutoff+1) ]
    left_bins = model_bins[ left_mask_cutoff:(right_decent_stats_cutoff+1) ]

    # use the right- and left-half mean values for points with decent stats to determine whether each ratio has a left or right anchor point
    left_half_bins = decent_stats_indices[ 0:int(np.round(len(decent_stats_indices)/2)) ]
    right_half_bins = decent_stats_indices[ int(np.round(len(decent_stats_indices)/2)): ]
    has_left_anchor = []
    for ratio in [hist1/hist2, hist2/hist1]:
        
        if np.mean( ratio[left_half_bins] ) < np.mean( ratio[right_half_bins] ): # ratio has left anchor
            has_left_anchor.append(True)
            
        else: # ratio has right anchor
            has_left_anchor.append(False)
        
    # check that the two ratios have opposite anchor points as determined in this way. Otherwise, print a warning to the user.
    if not has_left_anchor.count(True) == has_left_anchor.count(False):
        print('Failed to unambiguously identify right and left anchor points! If the data has low statistics, kappa may not be extracted properly.')
    if has_left_anchor[0]: # hist1/hist2 has left anchor 
        anchor_bins_12 = left_bins
        anchor_bins_21 = right_bins
    else:
        anchor_bins_12 = right_bins
        anchor_bins_21 = left_bins
   
    for sample_index in range( len(all_samples) ):
    
        samplenow = put_fits_in_order(all_samples[sample_index], histbins, hist1, histbins, hist2) 
        
        [params,[fracs1,fracs2]] = get_params_and_fracs(samplenow) 
        
        fit1 = np.concatenate( (params, fracs1) )
        fit2 = np.concatenate( (params, fracs2) )

        ratio12 = [model_func(*fit1,x)/model_func(*fit2,x) for x in anchor_bins_12]
        ratio21 = [model_func(*fit2,x)/model_func(*fit1,x) for x in anchor_bins_21]
         
        kappa12now_arg = anchor_bins_12[np.argmin(ratio12)]
        kappa12now = np.min(ratio12)
            
        kappa21now_arg = anchor_bins_21[np.argmin(ratio21)]
        kappa21now = np.min(ratio21)

        ratio12_full = [model_func(*fit1,x)/model_func(*fit2,x) for x in model_bins] # used for plotting only
        ratio21_full = [model_func(*fit2,x)/model_func(*fit1,x) for x in model_bins] # used for plotting only
        ax1.plot(kappa12now_arg+min_bin,kappa12now,'ko',alpha=0.1)
        ax1.plot(model_bins+min_bin, ratio12_full,color='r',alpha=0.1)
        
        if sample_index==0:
            ax2.plot(kappa21now_arg+min_bin,kappa21now,'ko',alpha=0.1,label='extracted kappas')
            ax2.plot(model_bins+min_bin, ratio21_full, color='r', alpha=0.1,label='MCMC fits')
        else:
            ax2.plot(kappa21now_arg+min_bin,kappa21now,'ko',alpha=0.1)
            ax2.plot(model_bins+min_bin, ratio21_full, color='r', alpha=0.1)
        
        kappa12[sample_index] = kappa12now
        kappa21[sample_index] = kappa21now
                
    
    ax1.set_ylim((0,3))
    ax2.set_ylim((0,3))
    ax1.set_ylabel('hist1/hist2')
    ax2.set_ylabel('hist2/hist1')
    ax1.set_xlabel('Constituent multiplicity')
    ax2.set_xlabel('Constituent multiplicity')
    ax2.legend()
    fig.suptitle('MCMC Fit and Extracted Kappas')
    current_dir = Path.cwd()
    plt.savefig(current_dir / 'plots' / system / filelabel)
    
    return [kappa12, kappa21]


#####################################################################
# Functions to calculate topics and fractions from extracted kappas #
#####################################################################
    
def topic_and_err(p1, p1_errs, p2, p2_errs, kappa, kappa_errs):
    topic = (p1 - kappa*p2)/(1-kappa)
    topic_errs = np.sqrt((p1 - p2)**2 * kappa_errs**2 + (1 - kappa)**2 * (p1_errs**2 + kappa**2 * p2_errs**2)) / (1 - kappa)**2
    return topic, topic_errs

def calc_topics(p1, p1_errs, p2, p2_errs, kappa12, kappa21):
    topic1, topic1_err = topic_and_err(p1,p1_errs,p2,p2_errs,*kappa12)
    topic2, topic2_err = topic_and_err(p2,p2_errs,p1,p1_errs,*kappa21)
    return topic1, topic2_err, topic2, topic2_err

def calc_fracs(kappa12, kappa21):
    
    [k12, k12_err] = kappa12
    [k21, k21_err] = kappa21
    
    f1 = -(1-k12)/(-1+k12*k21)
    f2 = ((-1+k12)*k21)/(-1+k12*k21)
    f1_err = np.sqrt((k21 - 1)**2 * k12_err**2 + k12**2 * (k12 - 1)**2 * k21_err**2) / (k12 * k21 - 1)**2
    f2_err = np.sqrt(k21**2 * (k21 - 1)**2 * k12_err**2 + (k12 - 1)**2 * k21_err**2) / (k12 * k21 - 1)**2

    return [[f1, f1_err], [f2, f2_err]]

def calc_fracs_distribution(kappas):

    f1 = np.array([calc_fracs([kappas[0][i],0],[kappas[1][i],0])[0][0] for i in range(len(kappas[0]))])
    f2 = np.array([calc_fracs([kappas[0][i],0],[kappas[1][i],0])[1][0] for i in range(len(kappas[0]))])
    return [f1,f2]

#####################################################################
# Functions to plot topics and fractions ############################
#####################################################################

def plot_unnormalized_input(datum1, datum2, bins, filelabel, system):
    [[hist1_normalized, hist1_errs, hist1], hist1_n] = datum1
    [[hist2_normalized, hist2_errs, hist2], hist2_n] = datum2
    bins = range(min_bin, max_bin+1)
    histbins = get_mean(bins)

    plt.errorbar(histbins, hist1, hist1_errs*hist1_n, color='blue', label='histogram 1')
    plt.errorbar(histbins, hist2, hist2_errs*hist2_n, color='red',label='histogram 2')
    plt.xlabel('Constituent multiplicity')
    plt.ylabel('N')
    plt.legend()
    plt.xlim((min_bin,max_bin))
    plt.title('Input Histograms (Unnormalized)')
    current_dir = Path.cwd()
    plt.savefig(current_dir / 'plots' / system / (filelabel+'_inputs_unnormalized.png'))

def plot_topics(datum1, datum2, datumQ, datumG, bins, kappas, filelabel, system):
    [[hist1, hist1_errs, _], _] = datum1
    [[hist2, hist2_errs, _], _] = datum2
    [[histQ, histQ_errs, _], _] = datumQ
    [[histG, histG_errs, _], _] = datumG
    histbins = get_mean(bins)
    
    kappa12 = [np.mean(kappas[0]), np.std(kappas[0])]
    kappa21 = [np.mean(kappas[1]), np.std(kappas[1])]
    topic1, topic1_err, topic2, topic2_err = calc_topics(hist1, hist1_errs, hist2, hist2_errs, kappa12, kappa21) # t1 is [topic1, topic1errs], etc

    fig, ax = plt.subplots()
    
    ax.step(histbins,topic1,where='mid',color='blue',label='Topic 1')
    ax.fill_between(bins[1:], topic1-topic1_err, topic1+topic1_err, step='pre', color='blue', alpha=0.3)
    ax.step(histbins,topic2,where='mid',color='red', label='Topic 2')
    ax.fill_between(bins[1:], topic2-topic2_err, topic2+topic2_err, step='pre', color='red', alpha=0.3)

    ax.plot(get_mean(bins),histQ,color='k',label=r'$\gamma$+q')
    ax.plot(get_mean(bins),histG,color='k',linestyle='--',dashes=(5,5),label=r'$\gamma$+g')
    
    ax.set_xlim((bins[0],bins[-1]))
    ax.set_xlabel('Constituent multiplicity')
    ax.set_ylabel('Probability')
    ax.legend()
    plt.title('Resulting Topics')
    plt.tight_layout()
    
    current_dir = Path.cwd()
    fig.savefig(current_dir / 'plots' / system / (filelabel+'_topics.png'))


    # plot ratios
    dist_topic1g = distance.cosine(topic1, histG)
    # dist_topic1q = distance.cosine(topic1, histQ)
    dist_topic2g = distance.cosine(topic2, histG)
    # dist_topic2q = distance.cosine(topic2, histQ)

    # print(dist_topic1g, dist_topic1q, dist_topic2g, dist_topic2q)
    if dist_topic1g < dist_topic2g: # this means topic 1 is gluon and topic 2 is quark (gluon is easier to match)
        fig, ax = plt.subplots()
        ax.plot(get_mean(bins),np.divide(topic1, topic2), color='blue', label='Topic 1 / Topic 2')
        # ax.step(histbins,np.divide(topic1,topic2),where='mid',color='blue',label='Topic 1/Topic 2')
        ax.plot(get_mean(bins),np.divide(histG, histQ),color='k',label=r'$\gamma$+g / $\gamma$+q')
        save_plot(fig, ax, 'Constituent Multiplicity', 'Ratio', 'Resulting Topics Ratios (q/g)', 'ratiosqg', (bins[0],bins[-1]), (-20, 20), filelabel, system)

        fig, ax = plt.subplots()
        # ax.step(histbins,np.divide(topic2,topic1),where='mid',color='red',label='Topic 2/Topic 1')
        ax.plot(get_mean(bins),np.divide(topic2, topic1), color='red', label='Topic 2 / Topic 1')
        ax.plot(get_mean(bins),np.divide(histQ, histG),color='k',label=r'$\gamma$+q / $\gamma$+g')
        save_plot(fig, ax, 'Constituent Multiplicity', 'Ratio', 'Resulting Topics Ratios (g/q)', 'ratiosgq', (bins[0],bins[-1]), (-20, 20), filelabel, system)
    else:
        fig, ax = plt.subplots()
        # ax.step(histbins,np.divide(topic1,topic2),where='mid',color='red',label='Topic 1/Topic 2')
        ax.plot(get_mean(bins),np.divide(topic1, topic2), color='blue', label='Topic 1 / Topic 2')
        ax.plot(get_mean(bins),np.divide(histQ, histG),color='k',label=r'$\gamma$+q / $\gamma$+g')
        save_plot(fig, ax, 'Constituent Multiplicity', 'Ratio', 'Resulting Topics Ratios (q/g)', 'ratiosqg', (bins[0],bins[-1]), (-20, 20), filelabel, system)

        fig, ax = plt.subplots()
        # ax.step(histbins,np.divide(topic2,topic1),where='mid',color='red',label='Topic 2/Topic 1')
        ax.plot(get_mean(bins),np.divide(topic1, topic1), color='red', label='Topic 2 / Topic 1')
        ax.plot(get_mean(bins),np.divide(histG, histQ),color='k',label=r'$\gamma$+g / $\gamma$+q')
        save_plot(fig, ax, 'Constituent Multiplicity', 'Ratio', 'Resulting Topics Ratios (g/q)', 'ratiosgq', (bins[0],bins[-1]), (-20, 20), filelabel, system)

def save_plot(fig, ax, xlabel, ylabel, title, suffix, xlim, ylim, filelabel, system):
    ax.set_xlim(xlim)
    ax.set_xlabel(xlabel)
    ax.set_ylim(ylim)
    ax.set_ylabel(ylabel)
    ax.legend()
    plt.title(title)
    plt.tight_layout()

    current_dir = Path.cwd()
    print(current_dir)
    fig.savefig(current_dir / 'plots' / system / (filelabel+f'_{suffix}.png'))

def plot_fractions(kappas, filelabel, system):
    
    [f1,f2] = calc_fracs_distribution(kappas)
    
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(8,5))
    
    ax1.hist(x=f1, color='k', bins=20, alpha=0.3)
    ax1.hist(x=f1, color='k', bins=20, histtype='step')
    ax1.set_xlabel('fraction1')
    ax1.set_ylabel('Count')
    
    ax2.hist(x=f2, color='k', bins=20, alpha=0.3)
    ax2.hist(x=f2, color='k', bins=20, histtype='step')
    ax2.set_xlabel('fraction2')
    ax2.set_ylabel('Count')
    plt.tight_layout()

    plt.subplots_adjust(top=0.9)
    fig.suptitle('Fractions Histogram')
    
    current_dir = Path.cwd()
    fig.savefig(current_dir / 'plots' / system / (filelabel+'_fractions.png'))



#####################################################################
# Helper functions ##################################################
#####################################################################

def get_mean(mylist):
    
    return np.array( [(mylist[i]+mylist[i+1])/2 for i in range(0,len(mylist)-1)])


def get_square_diff(y1, y2):
    
    return np.sum( (y1-y2)**2 )
    
######################
## because the fit simultaneously describes two histograms with the same parameters but different sets of fractions,
## you don't know a priori which fractions describe which fit. Identify the fits with a histogram by using the smallest
## total squared difference between the fits and the histograms
#####################
def put_fits_in_order(theta, x1, hist1, x2, hist2):

    [params,[fracs1,fracs2]] = get_params_and_fracs(theta) 
        
    fitx = np.concatenate( (params, fracs1) )
    fity = np.concatenate( (params, fracs2) )
           
    diff_x1_y2 = get_square_diff( [model_func(*fitx,x) for x in x1], hist1 ) + get_square_diff( [model_func(*fity,x) for x in x2], hist2 )
    diff_x2_y1 = get_square_diff( [model_func(*fitx,x) for x in x2], hist2 ) + get_square_diff( [model_func(*fity,x) for x in x1], hist1 )        
   
    if diff_x1_y2 < diff_x2_y1:
        return np.concatenate( (params, fracs1, fracs2) )
    else:
        return np.concatenate( (params, fracs2, fracs1) )

def plot_hist(int_bins, orig_bins, func):
    
    func_vals = np.zeros( len(int_bins) )
    
    for i in range(len(int_bins)):
    
        b = int_bins[i]
        binind = np.digitize(b, orig_bins)-1

        func_vals[i] = func[binind]
        
    return func_vals
    
#####################################################################
#####################################################################
#####################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('system') # this should be string representing the csv file name, if csv file is ./data/150_pt100_qgfracs.csv, user should input 150_pt100_qgfracs
    parser.add_argument('sample1')
    parser.add_argument('sample2')
    parser.add_argument('nwalkers', type=int)
    parser.add_argument('nsamples', type=int)
    parser.add_argument('burn_in', type=int)
    parser.add_argument('nkappa', type=int)
    parser.add_argument('min_bin', type=int)
    parser.add_argument('max_bin', type=int)
    args = parser.parse_args()

    system = args.system
    sample1_label = args.sample1
    sample2_label = args.sample2
    nwalkers = args.nwalkers
    nsamples = args.nsamples
    burn_in = args.burn_in
    nkappa = args.nkappa
    min_bin = args.min_bin
    max_bin = args.max_bin

    if nkappa > (nsamples-burn_in)*nwalkers:
        print('number of times to try to sample kappa must be smaller than (nsamples-burn_in)*nwalkers')

    folder_name = system + '_' + sample1_label + '_' + sample2_label
    filelabel = folder_name + '_SN,N=4_'+str(nwalkers)+','+str(nsamples)+','+str(burn_in)
    # savelabel = filelabel+'_pt'+str(ptindex)

    samples = get_data(f'./data/{system}.csv', sample1_label, sample2_label)
    sample1, sample2, sample_quarks, sample_gluons = format_samples(samples, min_bin, max_bin)

    try:
        os.makedirs(f'./plots/{folder_name}')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    bins = range(min_bin, max_bin+1)
    plot_unnormalized_input(sample1, sample2, bins, filelabel, folder_name)

    # bins = range(max_bin-min_bin+1) # we will need to shift the axes
    # kappas_now = do_MCMC_and_get_kappa(
    #     sample1,
    #     sample2,
    #     min_bin,
    #     max_bin,
    #     filelabel,
    #     folder_name,
    #     nwalkers=nwalkers,
    #     nsamples=nsamples,
    #     burn_in=burn_in,
    #     nkappa=nkappa,
    #     variation_factor=1e-1, 
    #     trytimes=60000 if DO_MCMC else 100, # todo change
    #     bounds=[(0,50),(1,15),(-20,20),(0,50),(1,15),(-20,20),(0,50),(1,15),(-20,20),(0,50),(1,15),(-20,20),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1)],
    #     fit_init_point=[13,1.5,1.5,10,1.5,1.5,5,2,2,5,2,2,0.5,0.3,0.5,0.3,0.5,0.3]
    # )
    # with open('kappas_now.pkl', 'wb+') as f:
    #     pickle.dump(kappas_now, f)

    print('PLOTTING')

    with open('kappas_now.pkl', 'rb') as f:
        kappas_now = pickle.load(f)

    plot_fractions(kappas_now, filelabel, folder_name)
    plot_topics(sample1, sample2, sample_quarks, sample_gluons, bins, kappas_now, filelabel, folder_name)



    # parser = argparse.ArgumentParser(description='Make plot data for paper')
    # parser.add_argument('system')
    # parser.add_argument('ptindex', type=int)
    # parser.add_argument('nwalkers', type=int)
    # parser.add_argument('nsamples', type=int)
    # parser.add_argument('burn_in', type=int)
    # parser.add_argument('nkappa', type=int)
    # # make this more flexible so we can use non-int bins
    # # parser.add_argument('min_bin', type=int)
    # # parser.add_argument('max_bin', type=int)
    # args = parser.parse_args()

    # system = args.system
    # ptindex = args.ptindex
    # nwalkers = args.nwalkers
    # nsamples = args.nsamples
    # burn_in = args.burn_in
    # nkappa = args.nkappa
    # # min_bin = args.min_bin
    # # max_bin = args.max_bin
    
    # if nkappa > (nsamples-burn_in)*nwalkers:
    #     print('number of times to try to sample kappa must be smaller than (nsamples-burn_in)*nwalkers')
    # # if ptindex<0 or ptindex>=3:
    # #     print('Only valid ptindex values are 0, 1, or 2.')
    
    # if system=='PP':
    #     filename = 'PP_JEWEL_etamax1_constmult'
    # if system=='HI':
    #     filename = 'HI_JEWEL_etamax1_constmult_13invnbYJ'
    # else:
    #     filename = system

    # [indJJ, indYJ, indQ, indG] = list(range(0,4))

    # current_dir = Path.cwd()
    # # input_dir = current_dir / "inputs"
    # # kappas_dir = current_dir / "kappas"

    # file = open( current_dir / "inputs" / (filename+'.pickle'), 'rb')
    # datum = pickle.load(file)
    # file.close()
    # # print(datum)

    # filelabel = system+'_SN,N=4_'+str(nwalkers)+','+str(nsamples)+','+str(burn_in)
    # savelabel = filelabel+'_pt'+str(ptindex)

    # import os
    # try:
    #     os.makedirs(f'./plots/{system}')
    # except OSError as e:
    #     if e.errno != errno.EEXIST:
    #         raise

    # dataJJ = datum[indJJ][ptindex] if ptindex != -1 else datum[indJJ]
    # dataYJ = datum[indYJ][ptindex] if ptindex != -1 else datum[indYJ]

    # # print(len(dataJJ[0][0]), len(dataYJ[0][0]))
    
    # # quickly iterate through the data and get the bins
    # # for i in range(3):
    # #     dataJJ[0][i] = dataJJ[0][i][min_bin:max_bin]
    # #     dataYJ[0][i] = dataYJ[0][i][min_bin:max_bin]
    
    # print(len(dataJJ[0][0]), len(dataYJ[0][0]))

    # min_bin = 0
    # max_bin = len(dataJJ[0][0]) + 1

    # bins = range(int(min_bin),int(max_bin))   # 0 indexed 
    # kappas_now = do_MCMC_and_get_kappa(
    #     dataJJ,
    #     dataYJ,  
    #     bins, 
    #     savelabel,
    #     system,
    #     nwalkers=nwalkers, 
    #     nsamples=nsamples, 
    #     burn_in=burn_in,
    #     nkappa=nkappa, 
    #     variation_factor=1e-1, 
    #     trytimes=60000 if DO_MCMC else 1000, # todo change
    #     bounds=[(0,50),(1,15),(-20,20),(0,50),(1,15),(-20,20),(0,50),(1,15),(-20,20),(0,50),(1,15),(-20,20),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1)],
    #     fit_init_point=[13,1.5,1.5,10,1.5,1.5,5,2,2,5,2,2,0.5,0.3,0.5,0.3,0.5,0.3]
    # )

    # file = open(current_dir / "kappas" / ('kappas_'+system+'_SN,N=4_'+str(nwalkers)+','+str(nsamples)+','+str(burn_in)+'_pt'+str(ptindex)+'.pickle'), 'wb')
    # pickle.dump([kappas_now], file)
    # file.close()

    # plot_fractions(kappas_now, filelabel, system)
    # plot_topics(datum[indJJ][ptindex], datum[indYJ][ptindex], datum[indQ][ptindex], datum[indG][ptindex], bins, kappas_now, filelabel, system)
    # # plot_topics(dataJJ, dataYJ, bins, kappas_now, filelabel, system)

