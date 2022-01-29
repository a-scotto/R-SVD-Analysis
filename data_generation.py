#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on August 22, 2019 at 15:12.

@author: a.scotto

Description:
"""
import os
import tqdm
import numpy
import json
import scipy.sparse
import scipy.linalg
import scipy.sparse.linalg

n = 1000
R = 10

# |----------------------------------|   Test Matrix Definition   |--------------------------------------| #
# |______________________________________________________________________________________________________| #

U, _ = scipy.linalg.qr(numpy.random.randn(n, n))
V, _ = scipy.linalg.qr(numpy.random.randn(n, n))

p = 0.5
step = numpy.ones(R)
singular_values = numpy.hstack([step, (numpy.arange(n-R) + 2)**(-p)])

A = U @ numpy.diag(singular_values) @ V.T


# |---------------------------------|   Experimental data vs. Samples   |--------------------------------| #
# |______________________________________________________________________________________________________| #

N = 100
POWER_SCHEME = False

# Form the different target low-rank truncations
temp = numpy.copy(singular_values)

temp[:5] = numpy.zeros(5)
A_perp_5 = scipy.sparse.diags(temp)

temp[:10] = numpy.zeros(10)
A_perp_10 = scipy.sparse.diags(temp)

temp[:15] = numpy.zeros(15)
A_perp_15 = scipy.sparse.diags(temp)

p_ = numpy.arange(7, 116)

meta_data = dict(singular_values=singular_values.tolist(), n_samples=p_.tolist(), target_rank=[5, 10, 15])

# Initialize the data
base = dict(single_pass=dict(Spectral=numpy.zeros((N, len(p_))), Frobenius=numpy.zeros((N, len(p_)))),
            power_scheme_1=dict(Spectral=numpy.zeros((N, len(p_))), Frobenius=numpy.zeros((N, len(p_)))),
            power_scheme_2=dict(Spectral=numpy.zeros((N, len(p_))), Frobenius=numpy.zeros((N, len(p_)))))

rank_5 = dict(single_pass=dict(Spectral=numpy.zeros((N, len(p_))), Frobenius=numpy.zeros((N, len(p_)))),
              power_scheme_1=dict(Spectral=numpy.zeros((N, len(p_))), Frobenius=numpy.zeros((N, len(p_)))),
              power_scheme_2=dict(Spectral=numpy.zeros((N, len(p_))), Frobenius=numpy.zeros((N, len(p_)))))

rank_10 = dict(single_pass=dict(Spectral=numpy.zeros((N, len(p_))), Frobenius=numpy.zeros((N, len(p_)))),
               power_scheme_1=dict(Spectral=numpy.zeros((N, len(p_))), Frobenius=numpy.zeros((N, len(p_)))),
               power_scheme_2=dict(Spectral=numpy.zeros((N, len(p_))), Frobenius=numpy.zeros((N, len(p_)))))

rank_15 = dict(single_pass=dict(Spectral=numpy.zeros((N, len(p_))), Frobenius=numpy.zeros((N, len(p_)))),
               power_scheme_1=dict(Spectral=numpy.zeros((N, len(p_))), Frobenius=numpy.zeros((N, len(p_)))),
               power_scheme_2=dict(Spectral=numpy.zeros((N, len(p_))), Frobenius=numpy.zeros((N, len(p_)))))

print(spectrum + ': ', end='')

for j, p in enumerate(p_):

    print(p, end='')

    for i in range(N):
        Omega = numpy.random.randn(n, p)

        # |----------------------------------------| Single-pass |---------------------------------------------|
        Z = A @ Omega
        Z, _ = scipy.linalg.qr(Z, mode='economic')

        # Compute the error matrices
        ZTA = (A @ Z).T
        E = A - Z @ ZTA

        ZTA = (A_perp_5 @ Z).T
        E_5 = A_perp_5 - Z @ ZTA

        ZTA = (A_perp_10 @ Z).T
        E_10 = A_perp_10 - Z @ ZTA

        ZTA = (A_perp_15 @ Z).T
        E_15 = A_perp_15 - Z @ ZTA

        # Store the corresponding error norms
        base['single_pass']['Spectral'][i, j] = numpy.linalg.norm(E, 2)
        base['single_pass']['Frobenius'][i, j] = numpy.linalg.norm(E, 'fro')

        rank_5['single_pass']['Spectral'][i, j] = numpy.linalg.norm(E_5, 2)
        rank_5['single_pass']['Frobenius'][i, j] = numpy.linalg.norm(E_5, 'fro')

        rank_10['single_pass']['Spectral'][i, j] = numpy.linalg.norm(E_10, 2)
        rank_10['single_pass']['Frobenius'][i, j] = numpy.linalg.norm(E_10, 'fro')

        rank_15['single_pass']['Spectral'][i, j] = numpy.linalg.norm(E_15, 2)
        rank_15['single_pass']['Frobenius'][i, j] = numpy.linalg.norm(E_15, 'fro')

        if POWER_SCHEME:
            # |----------------------------------------| Power Scheme 1 |------------------------------------------|
            Z = A @ (A.T @ Z)
            Z, _ = scipy.linalg.qr(Z, mode='economic')

            # Compute the error matrices
            E = A - Z @ (Z.T @ A)
            E_5 = A_perp_5 - Z @ (Z.T @ A_perp_5)
            E_10 = A_perp_10 - Z @ (Z.T @ A_perp_10)
            E_15 = A_perp_15 - Z @ (Z.T @ A_perp_15)

            # Store the corresponding error norms
            base['power_scheme_1']['Spectral'][i, j] = numpy.linalg.norm(E, 2)
            base['power_scheme_1']['Frobenius'][i, j] = numpy.linalg.norm(E, 'fro')**2

            rank_5['power_scheme_1']['Spectral'][i, j] = numpy.linalg.norm(E_5, 2)
            rank_5['power_scheme_1']['Frobenius'][i, j] = numpy.linalg.norm(E_5, 'fro')**2

            rank_10['power_scheme_1']['Spectral'][i, j] = numpy.linalg.norm(E_10, 2)
            rank_10['power_scheme_1']['Frobenius'][i, j] = numpy.linalg.norm(E_10, 'fro')**2

            rank_15['power_scheme_1']['Spectral'][i, j] = numpy.linalg.norm(E_15, 2)
            rank_15['power_scheme_1']['Frobenius'][i, j] = numpy.linalg.norm(E_15, 'fro')**2


            # |----------------------------------------| Power Scheme 2 |------------------------------------------|
            Z = A @ (A.T @ Z)
            Z, _ = scipy.linalg.qr(Z, mode='economic')

            # Compute the error matrices
            E = A - Z @ (Z.T @ A)
            E_5 = A_perp_5 - Z @ (Z.T @ A_perp_5)
            E_10 = A_perp_10 - Z @ (Z.T @ A_perp_10)
            E_15 = A_perp_15 - Z @ (Z.T @ A_perp_15)

            # Store the corresponding error norms
            base['power_scheme_2']['Spectral'][i, j] = numpy.linalg.norm(E, 2)
            base['power_scheme_2']['Frobenius'][i, j] = numpy.linalg.norm(E, 'fro')**2

            rank_5['power_scheme_2']['Spectral'][i, j] = numpy.linalg.norm(E_5, 2)
            rank_5['power_scheme_2']['Frobenius'][i, j] = numpy.linalg.norm(E_5, 'fro')**2

            rank_10['power_scheme_2']['Spectral'][i, j] = numpy.linalg.norm(E_10, 2)
            rank_10['power_scheme_2']['Frobenius'][i, j] = numpy.linalg.norm(E_10, 'fro')**2

            rank_15['power_scheme_2']['Spectral'][i, j] = numpy.linalg.norm(E_15, 2)
            rank_15['power_scheme_2']['Frobenius'][i, j] = numpy.linalg.norm(E_15, 'fro')**2

    print(', ', end='')

    1/0

# Convert to list for JSON
base['single_pass']['Spectral'] = base['single_pass']['Spectral'].tolist()
base['power_scheme_1']['Spectral'] = base['power_scheme_1']['Spectral'].tolist()
base['power_scheme_2']['Spectral'] = base['power_scheme_2']['Spectral'].tolist()

base['single_pass']['Frobenius'] = base['single_pass']['Frobenius'].tolist()
base['power_scheme_1']['Frobenius'] = base['power_scheme_1']['Frobenius'].tolist()
base['power_scheme_2']['Frobenius'] = base['power_scheme_2']['Frobenius'].tolist()


rank_5['single_pass']['Spectral'] = rank_5['single_pass']['Spectral'].tolist()
rank_5['power_scheme_1']['Spectral'] = rank_5['power_scheme_1']['Spectral'].tolist()
rank_5['power_scheme_2']['Spectral'] = rank_5['power_scheme_2']['Spectral'].tolist()

rank_5['single_pass']['Frobenius'] = rank_5['single_pass']['Frobenius'].tolist()
rank_5['power_scheme_1']['Frobenius'] = rank_5['power_scheme_1']['Frobenius'].tolist()
rank_5['power_scheme_2']['Frobenius'] = rank_5['power_scheme_2']['Frobenius'].tolist()


rank_10['single_pass']['Spectral'] = rank_10['single_pass']['Spectral'].tolist()
rank_10['power_scheme_1']['Spectral'] = rank_10['power_scheme_1']['Spectral'].tolist()
rank_10['power_scheme_2']['Spectral'] = rank_10['power_scheme_2']['Spectral'].tolist()

rank_10['single_pass']['Frobenius'] = rank_10['single_pass']['Frobenius'].tolist()
rank_10['power_scheme_1']['Frobenius'] = rank_10['power_scheme_1']['Frobenius'].tolist()
rank_10['power_scheme_2']['Frobenius'] = rank_10['power_scheme_2']['Frobenius'].tolist()


rank_15['single_pass']['Spectral'] = rank_15['single_pass']['Spectral'].tolist()
rank_15['power_scheme_1']['Spectral'] = rank_15['power_scheme_1']['Spectral'].tolist()
rank_15['power_scheme_2']['Spectral'] = rank_15['power_scheme_2']['Spectral'].tolist()

rank_15['single_pass']['Frobenius'] = rank_15['single_pass']['Frobenius'].tolist()
rank_15['power_scheme_1']['Frobenius'] = rank_15['power_scheme_1']['Frobenius'].tolist()
rank_15['power_scheme_2']['Frobenius'] = rank_15['power_scheme_2']['Frobenius'].tolist()

print('Done.')

data = [meta_data, base, rank_5, rank_10, rank_15, base_k, rank_k]

with open('Data_versus_samples.json', 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)



# |-------------------------------|   Experimental data vs. Target Rank   |------------------------------| #
# |______________________________________________________________________________________________________| #

p_MAX = 102

base_k = dict(Spectral=numpy.zeros((N, p_MAX - 2)), Frobenius=numpy.zeros((N, p_MAX - 2)))
rank_k = dict(Spectral=numpy.zeros((N, p_MAX - 2)), Frobenius=numpy.zeros((N, p_MAX - 2)))

base_Fro, base_Spec, Z_i = list(), list(), list()
for i in range(N):
    Omega = numpy.random.randn(n, p_MAX)

    Z, _ = scipy.linalg.qr(A @ Omega, mode='economic')
    Z_i.append(Z)

    ZT_A = (A @ Z).T
    E = A - Z @ ZT_A

    base_Spec.append(numpy.linalg.norm(E, 2))
    base_Fro.append(numpy.linalg.norm(E, 'fro'))

for k in tqdm.tqdm(range(p_MAX - 2)):
    singular_values[k] = 0
    A_perp_k = scipy.sparse.diags(singular_values)

    for i, Zi in enumerate(Z_i):
        ZTA = (A_perp_k @ Zi).T
        E_k = A_perp_k - Zi @ ZTA

        rank_k['Spectral'][i, k] = base_Spec[i] - numpy.linalg.norm(E_k, 2)
        rank_k['Frobenius'][i, k] = base_Fro[i] - numpy.linalg.norm(E_k, 'fro')

rank_k['Spectral'] = rank_k['Spectral'].tolist()
rank_k['Frobenius'] = rank_k['Frobenius'].tolist()

data = rank_k

with open('Data_versus_rank.json', 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)