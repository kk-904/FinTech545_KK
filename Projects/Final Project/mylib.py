import pandas as pd
import numpy as np
import math
import scipy.stats as stats
import scipy.optimize as opt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
from statsmodels.miscmodels.tmodel import TLinearModel
from scipy.stats import norm, t, skewnorm, norminvgauss


def corr(df, skipMissing=False):
    if skipMissing:
        return df.dropna().corr()
    return df.corr()


def cov(df, skipMissing=False):
    if skipMissing:
        return df.dropna().cov()
    return df.cov()


def _populateWeights(n, lambda_):
    tw = 0.0
    w = np.zeros(n)
    for i in range(n):
        w[i] = (1 - lambda_) * lambda_**i
        tw += w[i]
    w /= tw
    return w


def corrEW(df, lambda_):
    covMat = covEW(df, lambda_)
    stdDev = np.sqrt(np.diag(covMat))
    result = covMat / np.outer(stdDev, stdDev)
    return result


def covEW(df, lambda_):
    weights = _populateWeights(df.shape[0], lambda_)
    weights = weights[::-1]
    result = pd.DataFrame(
        np.cov(df.values.T, aweights=weights, ddof=0),
        index=df.columns,
        columns=df.columns,
    )
    return result


def covEW2(df, lambdaVar, lambdaCorr):
    covMat = covEW(df, lambdaVar)
    stdDev = np.sqrt(np.diag(covMat))
    corrMat = corrEW(df, lambdaCorr).values
    result = pd.DataFrame(
        np.diag(stdDev) @ corrMat @ np.diag(stdDev),
        index=df.columns,
        columns=df.columns,
    )
    return result


def _applyToCov(df, func):
    stdDev = np.sqrt(np.diag(df))
    corrMat = df / np.outer(stdDev, stdDev)
    corrMat = func(corrMat).values
    result = np.diag(stdDev) @ corrMat @ np.diag(stdDev)
    result = pd.DataFrame(result, index=df.columns, columns=df.columns)
    return result


def covNearPSD(df, epsilon=0.0):
    return _applyToCov(df, lambda x: corrNearPSD(x, epsilon))


def corrNearPSD(df, epsilon=0.0):
    eigVal, eigVec = np.linalg.eigh(df)
    val = np.maximum(eigVal, epsilon)
    vec = eigVec
    T = 1 / (np.multiply(vec, vec) @ val.T)
    T = np.sqrt(np.diag(np.array(T).reshape(df.shape[1])))
    B = T @ vec @ np.diag(np.sqrt(val))
    result = B @ B.T
    result = pd.DataFrame(result, index=df.columns, columns=df.columns)
    return result


def covHigham(df, epsilon=0.0, limit=1000):
    return _applyToCov(df, lambda x: corrHigham(x, epsilon, limit))


def corrHigham(df, epsilon=0.0, limit=1000):
    def _getPs(A, W, epsilon):
        sqrtW = np.sqrt(W)
        eigen_vals, eigen_vec = np.linalg.eig(sqrtW @ A @ sqrtW)
        Q = eigen_vec
        max_diag = np.diag(np.maximum(eigen_vals, epsilon))
        A_plus = Q @ max_diag @ Q.T
        sqrtW_i = np.linalg.inv(sqrtW)
        return sqrtW_i @ A_plus @ sqrtW_i

    def _getPu(A, W):
        Aret = A.copy()
        Aret[W > 0] = W[W > 0]
        return Aret

    m = df.shape[1]
    W = np.identity(m)
    # W is the matrix used for the norm (assumed to be Identity matrix here)
    # the algorithm should work for any diagonal W
    deltaS = np.zeros((m, m))
    Yk = df.values
    for i in range(0, limit):
        Rk = Yk - deltaS
        Xk = _getPs(Rk, W, epsilon)
        deltaS = Xk - Rk
        Y_next = _getPu(Xk, W)
        diff_norm = np.linalg.norm(Y_next - Yk, ord="fro")
        if diff_norm < epsilon:
            break  # Stop early if the change is small
        Yk = Y_next  # Update for next iteration
    result = pd.DataFrame(Yk, index=df.columns, columns=df.columns)
    return result

def cholPSD(df):
    L = np.linalg.cholesky(df.values)
    return pd.DataFrame(L, index=df.columns, columns=df.columns)

def _isSemiDefinitePositive(x, threshold=1e-8):
    return np.all(np.linalg.eigvals(x) >= threshold)

def normalSimulation(mean, cov, n, fix="near_psd"):
    m = mean.shape[0]
    Z = np.random.randn(m, n)
    if _isSemiDefinitePositive(cov):
        L = cholPSD(cov).values
    else:
        if fix == "near_psd":
            cov = covNearPSD(cov)
        elif fix == "higham":
            cov = covHigham(cov)
        L = cholPSD(cov).values 
    samples = (L @ Z).T + mean
    return pd.DataFrame(samples, columns=cov.columns)

def pcaSimulation(mean, cov, n_sim=100000, explained_var = None):
    vals, vecs = np.linalg.eig(cov)
    idx = vals.argsort()[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]
    tv = np.sum(vals)
    if explained_var is not None:
        total_var = 0
        n_elements = 0
        for val in vals:
            total_var += val
            n_elements += 1
            if total_var / tv >= explained_var:
                break
        vals = vals[:n_elements]
        vecs = vecs[:, :n_elements]
    else:
        vals = vals[vals > 1e-8]
        vecs = vecs[:, :len(vals)]
    B = vecs @ np.diag(np.sqrt(vals))
    r = np.random.randn(n_sim, len(vals))
    return pd.DataFrame(r @ B.T + mean, columns=cov.columns)
    
def arithmetricReturn(df):
    if 'Date' in df.columns:
        date_col = df['Date']
        result = df.drop(columns=['Date']).pct_change().dropna()
        result.insert(0, 'Date', date_col.iloc[1:].values)
        result.reset_index(drop=True, inplace=True)
        return result
    result = df.pct_change().dropna()
    return result.reset_index(drop=True, inplace=True)

def logReturn(df):
    if 'Date' in df.columns:
        date_col = df['Date']
        result = np.log(df.drop(columns=['Date']) / df.drop(columns=['Date']).shift(1)).dropna()
        result.insert(0, 'Date', date_col.iloc[1:].values)
        result.reset_index(drop=True, inplace=True)
        return result
    result = np.log(df / df.shift(1)).dropna()
    return result.reset_index(drop=True, inplace=True)

def fitNormal(data):
    mu = data.mean()
    sigma = data.std(ddof=1)
    return (mu, sigma)

def fitT(data):
    model = stats.t.fit(data)
    nu = model[0]
    mu = model[1]
    sigma = model[2]
    return (nu, mu, sigma)

def treg(df):
    y = df["y"].values
    X = df.copy().drop(columns=["y"]).values
    X = sm.add_constant(X)
    model = TLinearModel(y, X)
    result = model.fit()
    resid = y - result.predict()
    scale = result.params[-1]
    nu = result.params[-2]
    weights = (nu + 1) / (nu + (resid / scale) ** 2)
    mu = (resid * weights).mean()
    return np.concatenate([result.params, [mu]])

    

def var(df, alpha=0.05, dist="norm"):
    if dist == "norm":
        model = fitNormal(df)
        mu = model[0]
        sigma = model[1]
        return -stats.norm.ppf(alpha, mu, sigma)
    elif dist == "t":
        model = fitT(df)
        mu = model[1]
        sigma = model[2]
        nu = model[0]
        return -stats.t.ppf(alpha, nu, mu, sigma)
    elif dist == "skewnorm":
        params = stats.skewnorm.fit(df)
        a, loc, scale = params
        var = stats.skewnorm.ppf(alpha, a, loc=loc, scale=scale)
        # es = -loc + (scale * stats.skewnorm.pdf(var, alpha, loc=loc, scale=scale)) / alpha
        return -var
    elif dist == "norminvgauss":
        params = stats.norminvgauss.fit(df)
        a, b, loc, scale = params
        var = stats.norminvgauss.ppf(alpha, a, b, loc=loc, scale=scale)
        return -var
    elif dist == "sim":
        n_sim = 100000
        sim_data = np.random.choice(df.iloc[:, 0], n_sim, replace=True)
        return -np.quantile(sim_data, alpha)

    
def es(df, alpha=0.05, dist="norm"):
    if dist == "norm":
        model = fitNormal(df)
        mu = model[0]
        sigma = model[1]
        return -mu + sigma * stats.norm.pdf(stats.norm.ppf(alpha)) / alpha
    elif dist == "t":
        model = fitT(df)
        mu = model[1]
        sigma = model[2]
        nu = model[0]
        return -mu + (sigma * (nu+stats.t.ppf(alpha, nu) ** 2) / (nu-1)) * stats.t.pdf(stats.t.ppf(alpha, nu), nu) / alpha
    elif dist == "skewnorm":
        params = stats.skewnorm.fit(df)
        a, loc, scale = params
        n_samples = 100000
        samples = stats.skewnorm.rvs(a, loc=loc, scale=scale, size=n_samples)
        var = np.quantile(samples, alpha)
        es_samples = samples[samples < var].mean()
        return -es_samples
    elif dist == "norminvgauss":
        params = stats.norminvgauss.fit(df)
        a, b, loc, scale = params
        n_samples = 100000
        samples = stats.norminvgauss.rvs(a, b, loc=loc, scale=scale, size=n_samples)
        var = np.quantile(samples, alpha)
        es_samples = samples[samples < var].mean()
        return -es_samples
    elif dist == "sim":
        model = fitNormal(df)
        mu = model[0]
        sigma = model[1]
        n_samples = 100000
        samples = np.random.normal(mu, sigma, n_samples)
        var = samples.quantile(alpha)
        es_samples = samples[samples < var].mean()
        return -es_samples
    
def varesSimCopula(pf, ret, alpha=0.05, n_sim=100000):
    ret = ret - ret.mean()
    fitted_models = {}
    for _, row in pf.iterrows():
        if row['dist'] == 'Normal':
            fitted_models[row['stock']] = ('norm', fitNormal(ret[row['stock']]))
        elif row['dist'] == 'T':
            fitted_models[row['stock']] = ('t', fitT(ret[row['stock']]))
    u = pd.DataFrame(columns = pf['stock'])
    for stock in pf['stock']:
        if fitted_models[stock][0] == 'norm':
            u[stock] = stats.norm.cdf(ret[stock], scale = fitted_models[stock][1][1])
        elif fitted_models[stock][0] == 't':
            u[stock] = stats.t.cdf(ret[stock], fitted_models[stock][1][0], scale = fitted_models[stock][1][2])
    r = u.corr()
    copula = stats.multivariate_normal(mean=np.zeros(pf.shape[0]), cov=r)
    sim_u = pd.DataFrame(stats.norm.cdf(copula.rvs(size=n_sim)), columns=pf['stock'])
    sim_ret = pd.DataFrame(columns=pf['stock'])
    for stock in pf['stock']:
        if fitted_models[stock][0] == 'norm':
            sim_ret[stock] = stats.norm.ppf(sim_u[stock], scale = fitted_models[stock][1][1])
        elif fitted_models[stock][0] == 't':
            sim_ret[stock] = stats.t.ppf(sim_u[stock], fitted_models[stock][1][0], scale = fitted_models[stock][1][2])
    sim_pf_value = sim_ret @ (pf['holding'] * pf['price']).values
    pf_value = (pf['holding'] * pf['price']).sum()
    pf_var = -np.quantile(sim_pf_value, alpha)
    pf_es = -sim_pf_value[sim_pf_value <= np.quantile(sim_pf_value, alpha)].mean()
    stock_var = {}
    stock_es = {}
    stock_var_pct = {}
    stock_es_pct = {}
    for stock in pf['stock']:
        sim_stock_val = sim_ret[stock] * pf.loc[pf['stock'] == stock, 'holding'].values[0] * pf.loc[pf['stock'] == stock, 'price'].values[0]
        stock_value = pf.loc[pf['stock'] == stock, 'holding'].values[0] * pf.loc[pf['stock'] == stock, 'price'].values[0]
        stock_var[stock] = -np.quantile(sim_stock_val, alpha)
        stock_es[stock] = -sim_stock_val[sim_stock_val <= np.quantile(sim_stock_val, alpha)].mean()
        stock_var_pct[stock] = stock_var[stock] / stock_value
        stock_es_pct[stock] = stock_es[stock] / stock_value
    
    var_es_df = pd.DataFrame({
        'VaR': [stock_var[stock] for stock in pf['stock']],
        'ES': [stock_es[stock] for stock in pf['stock']],
        'VaR_Pct': [stock_var_pct[stock] for stock in pf['stock']],
        'ES_Pct': [stock_es_pct[stock] for stock in pf['stock']]
    }, index=pf['stock'])
    var_es_df.loc['Total'] = [pf_var, pf_es, pf_var / pf_value, pf_es / pf_value]
    return var_es_df
    
def risk_parity(df, risk_budget):
    cov_matrix = cov(df)
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    weights = inv_cov_matrix @ risk_budget
    weights /= np.sum(weights)
    return pd.Series(weights, index=df.columns)