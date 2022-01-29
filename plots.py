#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on August 22, 2019 at 15:12.

@author: a.scotto

Description:
"""
import os
import json
import numpy
import scipy.sparse

from numpy import sqrt as sqrt
from matplotlib import pyplot, rcParams, cm

rcParams.update({'axes.grid': True})
rcParams.update({'axes.axisbelow': True})
rcParams.update({'axes.facecolor': '#ffffff'})
rcParams.update({'axes.titlesize': 16})
rcParams.update({'axes.labelsize': 16})

rcParams.update({'grid.color': '#AAAAAA'})
rcParams.update({'xtick.labelsize': 14})
rcParams.update({'ytick.labelsize': 14})
rcParams.update({'legend.fontsize': 14})

rcParams.update({'lines.linewidth': 1})
rcParams.update({'lines.markersize': 8})
rcParams.update({'markers.fillstyle': 'none'})

rcParams.update({'legend.facecolor': '#ffffff'})
rcParams.update({'text.usetex': True})
rcParams.update( {'text.latex.preamble' : r'\usepackage{amsfonts}'})


# |----------------------------------| EXPECTATION Constants |----------------------------------| #
def C2_exp(sigma, choice, p, k, q):

    Gamma_k_inv = scipy.sparse.diags(1 / sigma[:k]**(4*q+2))

    if choice == 'Sigma_k':
        X =  scipy.sparse.diags(sigma[:k])
    elif choice == 'I_k':
        X = numpy.eye(k)
    elif choice == 'Sigma_k_hat':
        X =  scipy.sparse.diags(sqrt(sigma[:k]**2 - sigma[k]**2))

    Gamma_k_hat = (X @ Gamma_k_inv @ X).diagonal()
    Gamma_perp =  sigma[k:]**(4*q+2)

    term_1 = sqrt(Gamma_perp[0] * numpy.sum(Gamma_k_hat) / (p - k - 1))
    term_2 = sqrt(numpy.max(Gamma_k_hat) * numpy.sum(Gamma_perp) * p) * numpy.e / (p - k)

    return term_1 + term_2

def CF_exp(sigma, choice, p, k, q):

    Gamma_k_inv = scipy.sparse.diags(1 / sigma[:k]**(4*q+2))

    if choice == 'Sigma_k':
        X =  scipy.sparse.diags(sigma[:k])
    elif choice == 'I_k':
        X = numpy.eye(k)
    elif choice == 'Sigma_k_hat':
        X =  scipy.sparse.diags(sqrt(sigma[:k]**2 - sigma[k]**2))

    Gamma_k_hat = (X @ Gamma_k_inv @ X).diagonal()
    Gamma_perp =  sigma[k:]**(4*q+2)

    C = numpy.sum(Gamma_perp) * numpy.sum(Gamma_k_hat) / (p - k - 1)

    return C


# |----------------------------------| PROBABILITY Constants |----------------------------------| #
def C2_prob(sigma, choice, p, k, q):

    Gamma_k_inv = scipy.sparse.diags(1 / sigma[:k]**(4*q+2))

    if choice == 'Sigma_k':
        X =  scipy.sparse.diags(sigma[:k])
    elif choice == 'I_k':
        X = numpy.eye(k)
    elif choice == 'Sigma_k_hat':
        X =  scipy.sparse.diags(sqrt(sigma[:k]**2 - sigma[k]**2))

    Gamma_k_hat = (X @ Gamma_k_inv @ X).diagonal()
    Gamma_perp =  sigma[k:]**(4*q+2)

    C = sqrt(Gamma_perp[0] * numpy.max(Gamma_k_hat) * p) * numpy.e / (p - k)

    return C

def CF_prob(sigma, choice, p, k, q):

    Gamma_k_inv = scipy.sparse.diags(1 / sigma[:k]**(4*q+2))

    if choice == 'Sigma_k':
        X =  scipy.sparse.diags(sigma[:k])
    elif choice == 'I_k':
        X = numpy.eye(k)
    elif choice == 'Sigma_k_hat':
        X =  scipy.sparse.diags(sqrt(sigma[:k]**2 - sigma[k]**2))

    Gamma_k_hat = (X @ Gamma_k_inv @ X).diagonal()
    Gamma_perp =  sigma[k:]**(4*q+2)

    C = sqrt(Gamma_perp[0] * numpy.sum(Gamma_k_hat)) * sqrt(3*k / (p-k+1))

    return C



# |----------------------------------| EXPECTATION Bounds |----------------------------------| #
def phi(x):
    return x / sqrt(1 + x**2)

def HMT_Expectation(sigma, p, k, q, which):

    target = sigma[k] if which == 'Spectral' else numpy.sum(sigma[k:]**2)**0.5

    if which == 'Spectral':
        C = (C2_exp(sigma**(2*q+1), 'Sigma_k', p, k, 0) + target**(2*q+1))**(1 / (2*q+1))

    elif which == 'Frobenius':
        C = sqrt(CF_exp(sigma, 'Sigma_k', p, k, q) + target**2)

    return C

def Our_Expectation_OldMetric(sigma, p, k, q, which):

    target = sigma[k] if which == 'Spectral' else numpy.sum(sigma[k:]**2)**0.5

    if which == 'Spectral':
        # First solution, Gap-dependent
        a_k = C2_exp(sigma, 'Sigma_k_hat', p, k, q)
        c_k = C2_exp(sigma, 'I_k', p, k, q)

        option_1 = target + numpy.min([a_k, phi(c_k) * sqrt(sigma[0]**2 - sigma[k]**2)])

        # Second solution, Gap-independent
        sigma = sigma**(2*q+1)

        a_k = C2_exp(sigma, 'Sigma_k_hat', p, k, 0)
        c_k = C2_exp(sigma, 'I_k', p, k, 0)

        option_2 = (sigma[k] + numpy.min([a_k, phi(c_k) * sqrt(sigma[0]**2 - sigma[k]**2)]))**(1 / (2*q+1))

        # Mininum
        C = numpy.min([option_1, option_2])

    elif which == 'Frobenius':
        a_k = CF_exp(sigma, 'Sigma_k', p, k, q)
        c_k = CF_exp(sigma, 'I_k', p, k, q)

        C = sqrt(target**2 + numpy.min([a_k, sigma[0]**2 * k * phi(sqrt(k * c_k))**2]))

    return C

def Our_Expectation_NewMetric(sigma, p, k, q, which):

    if which == 'Spectral':
        a_k = C2_exp(sigma, 'Sigma_k', p, k, q)
        c_k = C2_exp(sigma, 'I_k', p, k, q)

        C = numpy.min([a_k, phi(c_k) * sigma[0]])

    elif which == 'Frobenius':
        a_k = CF_exp(sigma, 'Sigma_k', p, k, q)
        c_k = CF_exp(sigma, 'I_k', p, k, q)

        C = numpy.min([sqrt(a_k), sigma[0] * sqrt(k) * phi(sqrt(c_k / k))])

    return C


# |----------------------------------| PROBABILITY Bounds |----------------------------------| #
def HMT_Probability(sigma, p, k, q, which):

    alpha = 0.5
    u = sqrt(-2 * numpy.log(alpha * delta))
    t = ((1 - alpha) * 2 / delta)**(1 / (p - k))

    if which == 'Spectral':
        term_1 = (1 + t*sqrt(3*k / (p-k+1))) * sqrt(numpy.sum(sigma[k:]))
        term_2 = u*t * numpy.e * sqrt(p) / (p-k+1) * sigma[k]

        C = term_1 + term_2

    elif which == 'Frobenius':
        term_1 = (1 + t*sqrt(3*k / (p-k+1))) * sigma[k]
        term_2 = t * numpy.e * sqrt(p) / (p-k+1) * sqrt(numpy.sum(sigma[k:]))
        term_3 = u*t * numpy.e * sqrt(p) / (p-k+1) * sigma[k]

        C = term_1 + term_2 + term_3

    return C

def Our_Probability_OldMetric(sigma, p, k, q, which):

    target = sigma[k] if which == 'Spectral' else numpy.sum(sigma[k:]**2)**0.5

    if which == 'Spectral':
        # First solution, Gap-dependent
        a_k = C2_exp(sigma, 'Sigma_k_hat', p, k, q)
        c_k = C2_exp(sigma, 'I_k', p, k, q)

        option_1 = target + numpy.min([a_k, phi(c_k) * sqrt(sigma[0]**2 - sigma[k]**2)])

        # Second solution, Gap-independent
        sigma = sigma**(2*q+1)

        a_k = C2_exp(sigma, 'Sigma_k_hat', p, k, 0)
        c_k = C2_exp(sigma, 'I_k', p, k, 0)

        option_2 = (sigma[k] + numpy.min([a_k, phi(c_k) * sqrt(sigma[0]**2 - sigma[k]**2)]))**(1 / (2*q+1))

        # Mininum
        C = numpy.min([option_1, option_2])

    elif which == 'Frobenius':
        a_k = CF_exp(sigma, 'Sigma_k', p, k, q)
        c_k = CF_exp(sigma, 'I_k', p, k, q)

        C = sqrt(target**2 + numpy.min([a_k, sigma[0]**2 * k * phi(sqrt(k * c_k))**2]))

    return C

def Our_Probability_NewMetric(sigma, p, k, q, which, delta=0.01):

    alpha = 0.5
    u = sqrt(-2 * numpy.log(alpha * delta))
    t = ((1 - alpha) * delta)**(-1 / (p - k))

    if which == 'Spectral':
        a_k = C2_exp(sigma, 'Sigma_k', p, k, q)
        b_k = C2_prob(sigma, 'Sigma_k', p, k, q)

        c_k = C2_exp(sigma, 'I_k', p, k, q)
        d_k = C2_prob(sigma, 'I_k', p, k, q)

        C = numpy.min([a_k + b_k * t*u, phi(c_k + d_k * t*u) * sigma[0]])

    elif which == 'Frobenius':
        a_k = CF_exp(sigma, 'Sigma_k', p, k, q)
        b_k = CF_prob(sigma, 'Sigma_k', p, k, q)

        c_k = CF_exp(sigma, 'I_k', p, k, q)
        d_k = CF_prob(sigma, 'I_k', p, k, q)

        C = numpy.min([sqrt(a_k) + b_k * t*u, sigma[0] * sqrt(k) * phi(sqrt(c_k / k) + d_k * t*u / sqrt(k))])

    return C


with open('Data_versus_samples.json') as file:
    data = json.load(file)

meta_data, base, rank_5, rank_10, rank_15, _, _ = data

settings_new_metric = [(5, rank_5), (10, rank_10), (15, rank_15)]
settings = [(5, rank_5), (10, rank_10), (15, rank_15)]
settings_power = [(5, rank_5), (10, rank_10), (15, rank_15)]

markers = ['o', 's', 'd', '^', 'x', '.']
colors = cm.get_cmap('tab10', 10)

norms = ['Spectral', 'Frobenius']

# Get the sequence of singular values and experimental number of samples
sigma = numpy.asarray(meta_data['singular_values'])
p_ = numpy.asarray(meta_data['n_samples'])

PROBA = False

for which in norms:

    # Behavior of the bounds vs. Empirical Data
    for k, rank in settings_new_metric:
        pyplot.figure()

        mark_every = numpy.hstack([[2, 6, 10], 5 * numpy.arange(18) + 15]) - 2
        p_th = numpy.arange(99) + k + 2

        # Experimental Data
        data = numpy.asarray(base['single_pass'][which]) - numpy.asarray(rank['single_pass'][which])
        average, stdev = numpy.mean(data, axis=0), numpy.std(data, axis=0)

        average = average[(k-5):-(15-k)] if k != 15 else average[10:]
        stdev = stdev[(k-5):-(15-k)] if k != 15 else stdev[10:]
        p_expe = p_[(k-5):-(15-k)] if k != 15 else p_[10:]

        pyplot.plot(p_expe - k, average, label='Empirical mean error', clip_on=False,
                    color='k', ls='-', marker=markers[0], mfc='w', lw=1, markevery=mark_every)

        pyplot.fill_between(p_expe - k, average - stdev, average + stdev, clip_on=False, color='k', alpha=0.1)

        # Theoretical Expectation Bound
        our_bound = [Our_Expectation_NewMetric(sigma, p, k, 0, which) for p in p_th]
        pyplot.plot(p_th - k, our_bound, label='Our bound', clip_on=False, color='k', ls='--', marker=markers[1], markevery=mark_every)

        # Theoretical Probability Bound
        if PROBA:
            delta_ = [0.25, 0.05, 0.01]

            our_bound_1 = [Our_Probability_NewMetric(sigma, p, k, 0, which, delta=delta_[0]) for p in p_th]
            our_bound_2 = [Our_Probability_NewMetric(sigma, p, k, 0, which, delta=delta_[1]) for p in p_th]
            our_bound_3 = [Our_Probability_NewMetric(sigma, p, k, 0, which, delta=delta_[2]) for p in p_th]

            pyplot.plot(p_th - k, our_bound_1, label='$\mathbb{P}($failure$)=' + str(delta_[0]) + '$', clip_on=False,
                        color='k', ls=':', marker=markers[2], markevery=mark_every)

            pyplot.plot(p_th - k, our_bound_2, label='$\mathbb{P}($failure$)=' + str(delta_[1]) + '$', clip_on=False,
                        color='k', ls='--', marker=markers[3], markevery=mark_every)

            pyplot.plot(p_th - k, our_bound_3, label='$\mathbb{P}($failure$)=' + str(delta_[2]) + '$', clip_on=False,
                        color='k', ls='-.', marker=markers[4], markevery=mark_every)

        pyplot.yscale('log')

        pyplot.xlabel('Oversampling $\\varrho(p) = p-k$.')
        pyplot.xlim(p_th[0] - k, p_th[-1] - k)
        pyplot.xticks([2, 20, 40, 60, 80, 100])
        pyplot.title('{} norm'.format(which))

        pyplot.legend()
        pyplot.savefig('OurBounds_' + which + '_rank' + str(k) + '.pdf', bbox_inches='tight')

    # Bounds as a function of k, for a fixed value p
    for a in [1]:
        pyplot.figure()

        p_max = 102
        k_axis = numpy.arange(1, p_max-1)

        step = p_max // 10
        mark_every = numpy.hstack([[1], step * numpy.arange(10) + step]) - 1

        with open('Data_versus_rank.json') as file:
            data = json.load(file)

        # Experimental Data
        data = numpy.asarray(data[which])[:, :p_max-2]
        average, stdev = numpy.mean(data, axis=0), numpy.std(data, axis=0)

        pyplot.plot(k_axis, average, label='Empirical mean error', clip_on=False,
                    color='k', ls='-', marker=markers[0], mfc='w', lw=1, markevery=mark_every)

        pyplot.fill_between(k_axis, average - stdev, average + stdev, clip_on=False, color='k', alpha=0.1)

        # Theoretical Expectation Bound
        our_bound = numpy.asarray([Our_Expectation_NewMetric(sigma, p_max, k, 0, which) for k in k_axis])
        pyplot.plot(k_axis, our_bound, label='Our bound', clip_on=False, color='k', ls='--', marker=markers[1], markevery=mark_every)

        # Theoretical Probability Bound
        if PROBA:
            delta_ = [0.25, 0.05, 0.01]

            our_bound_1 = numpy.asarray([Our_Probability_NewMetric(sigma, p_max + 1, k, 0, which, delta=delta_[0]) for k in k_axis])
            our_bound_2 = numpy.asarray([Our_Probability_NewMetric(sigma, p_max + 1, k, 0, which, delta=delta_[1]) for k in k_axis])
            our_bound_3 = numpy.asarray([Our_Probability_NewMetric(sigma, p_max + 1, k, 0, which, delta=delta_[2]) for k in k_axis])

            pyplot.plot(k_axis, our_bound_1, label='$\mathbb{P}($failure$)=' + str(delta_[0]) + '$',
                        clip_on=False, color='k', ls=':', marker=markers[2], markevery=mark_every)

            pyplot.plot(k_axis, our_bound_2, label='$\mathbb{P}($failure$)=' + str(delta_[1]) + '$',
                        clip_on=False, color='k', ls='--', marker=markers[3], markevery=mark_every)

            pyplot.plot(k_axis, our_bound_3, label='$\mathbb{P}($failure$)=' + str(delta_[2]) + '$',
                        clip_on=False, color='k', ls='-.', marker=markers[4], markevery=mark_every)

        pyplot.yscale('log')

        pyplot.xlabel('Target rank $k$.')
        pyplot.xlim(k_axis[0], k_axis[-1])
        pyplot.xticks(mark_every[::2] + 1)
        pyplot.title('{} norm'.format(which))

        pyplot.legend()
        pyplot.savefig('Minimum_k_' + which + '_pMax' + str(p_max) + '.pdf', bbox_inches='tight')

    # Comparaison Single-Pass with Halko, Martinsson & Tropp
    for k, rank in settings:
        pyplot.figure()

        normalization = sigma[k] if which == 'Spectral' else sqrt(numpy.sum(sigma[k:]**2))
        mark_every = numpy.hstack([[2, 11, 20], 8 * numpy.arange(11) + 20]) - 2
        p_th = numpy.arange(99) + k + 2

        # Theoretical bounds
        hmt_bound = [HMT_Expectation(sigma, p, k, 0, which) for p in p_th]
        our_bound = numpy.asarray([Our_Expectation_NewMetric(sigma, p, k, 0, which) for p in p_th])

        our_bound = sqrt(our_bound**2 + normalization**2) if which == 'Frobenius' else our_bound + normalization

        pyplot.plot(p_th - k, hmt_bound, label='HMT bound', clip_on=False,
                    ls='-.', color='k', marker=markers[0], markevery=mark_every)

        pyplot.plot(p_th - k, our_bound, label='Our bound', clip_on=False,
                    ls='--', color='k', marker=markers[1], markevery=mark_every)

        if which == 'Spectral':
            our_bound_enhanced = [Our_Expectation_OldMetric(sigma, p, k, 0, which) for p in p_th]

            pyplot.plot(p_th - k, our_bound_enhanced, label='Our improved bound', clip_on=False,
                        ls=':', color='k', marker=markers[2], markevery=mark_every)

        pyplot.yscale('log')

        pyplot.xlabel('Oversampling $\\varrho(p) = p-k$.')
        pyplot.xlim(p_th[0] - k, p_th[-1] - k)
        pyplot.xticks([2, 20, 40, 60, 80, 100])
        pyplot.title('{} norm'.format(which))

        pyplot.legend()
        pyplot.savefig('Comparison_HMT_' + which + '_rank' + str(k) + '.pdf', bbox_inches='tight')

    # Comparison Power-Scheme with Halko, Martinsson & Tropp
    for k, rank in settings_power:
        pyplot.figure()

        mark_every = numpy.hstack([[2, 6, 10], 5 * numpy.arange(18) + 15]) - 2
        p_th = numpy.arange(99) + k + 2

        if which == 'Spectral':
            hmt_bound_0 = [HMT_Expectation(sigma, p, k, 0, which) for p in p_th]
            hmt_bound_1 = [HMT_Expectation(sigma, p, k, 1, which) for p in p_th]
            hmt_bound_2 = [HMT_Expectation(sigma, p, k, 2, which) for p in p_th]

        our_bound_0 = [Our_Expectation_OldMetric(sigma, p, k, 0, which) for p in p_th]
        our_bound_1 = [Our_Expectation_OldMetric(sigma, p, k, 1, which) for p in p_th]
        our_bound_2 = [Our_Expectation_OldMetric(sigma, p, k, 2, which) for p in p_th]

        if which == 'Spectral':
            pyplot.plot(p_th - k, hmt_bound_0, label='HMT bound $(q=0)$', clip_on=False,
                        color='k', ls='-', marker=markers[0], markevery=mark_every)

            pyplot.plot(p_th - k, hmt_bound_1, label='HMT bound $(q=1)$', clip_on=False,
                        color='k', ls=':', marker=markers[1], markevery=mark_every)

            pyplot.plot(p_th - k, hmt_bound_2, label='HMT bound $(q=2)$', clip_on=False,
                        color='k', ls='--', marker=markers[4], markevery=mark_every)


            pyplot.plot(p_th - k, our_bound_0, label='Our improved bound $(q=0)$', clip_on=False,
                        color='k', ls='-', marker=markers[2], markevery=mark_every)

            pyplot.plot(p_th - k, our_bound_1, label='Our improved bound $(q=1)$', clip_on=False,
                        color='k', ls=':', marker=markers[3], markevery=mark_every)

            pyplot.plot(p_th - k, our_bound_2, label='Our improved bound $(q=2)$', clip_on=False,
                        color='k', ls='--', mfc='k', fillstyle='full', marker=markers[5], markevery=mark_every)
        else:
            pyplot.plot(p_th - k, our_bound_0, label='Our bound $(q=0)$', clip_on=False,
                        color='k', ls='-', marker=markers[2], markevery=mark_every)

            pyplot.plot(p_th - k, our_bound_1, label='Our bound $(q=1)$', clip_on=False,
                        color='k', ls=':', marker=markers[3], markevery=mark_every)

            pyplot.plot(p_th - k, our_bound_2, label='Our bound $(q=2)$', clip_on=False,
                        color='k', ls='--', mfc='k', fillstyle='full', marker=markers[5], markevery=mark_every)

        pyplot.yscale('log')

        pyplot.xlabel('Oversampling $\\varrho(p) = p-k$.')
        pyplot.xlim(p_th[0] - k, p_th[-1] - k)
        pyplot.xticks([2, 20, 40, 60, 80, 100])
        pyplot.title('{} norm'.format(which))

        pyplot.legend()
        pyplot.savefig('PowerScheme_' + which + '_rank' + str(k) + '.pdf', bbox_inches='tight')

    pyplot.close()

# pyplot.show()