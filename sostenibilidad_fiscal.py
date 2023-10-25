import numpy as np
import pandas as pd


def dp_star_infinito(b_0, r, c):
    return - b_0 * (r-c)/(1+c)


def tau_star_infinito(b_0, r, c, ip_t, io_t, g_t, ip_m, io_m, g_m):
    w = g_m - ip_m - io_m
    z = g_t - ip_t - io_t - w
    [f,p] = [np.ones((z.shape[0]))*(1+r)/(1+c),np.array(list(range(1,z.shape[0]+1)))]
    S = (np.matrix( np.power(f,p) ) @ np.matrix(z).T)[0,0]
    return ((r-c)/(1+c))*(S+b_0) + w


def dp_star_finito(b_T, b_0, T, R, C):
	[T ,f]= [R.shape[0],(1+R)/(1+C)]
	M = np.ones((T,T))
	M = np.tril(M, -1)
	M = M.T*f
	M = M.T
	M[M == 0] = 1
	S = np.sum(np.prod(M,axis=0))
	return (b_T-np.prod(f)*b_0)/S


def tau_star_finito(b_T, b_0, R, C, G, IP, IO):
	[T ,f]= [R.shape[0],(1+R)/(1+C)]
	M = np.ones((T,T))
	M = np.tril(M, -1)
	M = M.T*f
	M = M.T
	M[M == 0] = 1
	fi = np.prod(M,axis=0)
	S = np.sum(fi)
	A = G - IP - IO
	SA = (np.matrix(A)  @ np.matrix(fi).T)[0,0]
	return (b_T-np.prod(f)*b_0-SA)/(-S)


def evolucion_deuda_tau(b_0, R, C, G, IP, IO, Tau):
    dp = G - IP - IO - Tau
    b_t_prime = []
    b_t_prime.append(b_0)
    for j in range(0,R.shape[0]):
        b_t_prime.append( ((1+R[j])/(1+C[j]))*b_t_prime[j] + dp[j] )
    return b_t_prime


def evolucion_deuda_dp(b_0, R, C, dp):
    b_t_prime = []
    b_t_prime.append(b_0)
    for j in range(0,R.shape[0]):
        b_t_prime.append( ((1+R[j])/(1+C[j]))*b_t_prime[j] + dp[j])
    return b_t_prime


def sensibilidad_dp_ss(b_0, grid, v_min, v_max):
    [interes,crec,M] = [np.linspace(v_min,v_max,grid),np.linspace(v_min,v_max,grid),np.zeros((grid,grid))]
    for k in range(0,crec.shape[0]):
        for j in range(0,interes.shape[0]):
            M[k,j] = dp_star_infinito(b_0,interes[j],crec[k])
    sen = pd.DataFrame(np.triu(M))
    [sen.columns,sen.index]= [interes,crec]
    [sen.columns,sen.index] = [sen.columns.rename('Inteŕes real'),sen.index.rename('Crecimiento real')]
    return sen


def sensibilidad_tau_ss(b_0, grid, v_min, v_max, ip_t, io_t, g_t, ip_m, io_m, g_m):
    [interes,crec,M] = [np.linspace(v_min,v_max,grid),np.linspace(v_min,v_max,grid),np.zeros((grid,grid))]
    for k in range(0,crec.shape[0]):
        for j in range(0,interes.shape[0]):
            M[k,j] = tau_star_infinito(b_0,interes[j],crec[k],ip_t,io_t,g_t,ip_m,io_m,g_m)
    sen = pd.DataFrame(np.triu(M))
    [sen.columns,sen.index]= [interes,crec]
    [sen.columns,sen.index] = [sen.columns.rename('Inteŕes real'),sen.index.rename('Crecimiento real')]
    return sen




