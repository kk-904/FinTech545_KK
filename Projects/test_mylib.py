import mylib
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats
import pandas.testing as pdt
import numpy.testing as npt


# def test_1_1():
#     sample = pd.read_csv("testfiles/data/test1.csv")
#     output = pd.read_csv("testfiles/data/testout_1.1.csv")
#     result = mylib.cov(sample, skipMissing=True)
#     npt.assert_almost_equal(result.values, output.values)
    
# def test_1_2():
#     sample = pd.read_csv("testfiles/data/test1.csv")
#     output = pd.read_csv("testfiles/data/testout_1.2.csv")
#     result = mylib.corr(sample, skipMissing=True)
#     npt.assert_almost_equal(result.values, output.values)
    
# def test_1_3():
#     sample = pd.read_csv("testfiles/data/test1.csv")
#     output = pd.read_csv("testfiles/data/testout_1.3.csv")
#     result = mylib.cov(sample, skipMissing=False)
#     npt.assert_almost_equal(result.values, output.values)
    
# def test_1_4():
#     sample = pd.read_csv("testfiles/data/test1.csv")
#     output = pd.read_csv("testfiles/data/testout_1.4.csv")
#     result = mylib.corr(sample, skipMissing=False)
#     npt.assert_almost_equal(result.values, output.values)
    
# def test_2_1():
#     sample = pd.read_csv("testfiles/data/test2.csv")
#     output = pd.read_csv("testfiles/data/testout_2.1.csv")
#     result = mylib.covEW(sample, .97)
#     npt.assert_almost_equal(result.values, output.values)
    
# def test_2_2():
#     sample = pd.read_csv("testfiles/data/test2.csv")
#     output = pd.read_csv("testfiles/data/testout_2.2.csv")
#     result = mylib.corrEW(sample, .94)
#     npt.assert_almost_equal(result.values, output.values)
    
# def test_2_3():
#     sample = pd.read_csv("testfiles/data/test2.csv")
#     output = pd.read_csv("testfiles/data/testout_2.3.csv")
#     result = mylib.covEW2(sample, .97, .94)
#     npt.assert_almost_equal(result.values, output.values)
    
# def test_3_1():
#     sample = pd.read_csv("testfiles/data/testout_1.3.csv")
#     output = pd.read_csv("testfiles/data/testout_3.1.csv")
#     result = mylib.covNearPSD(sample)
#     npt.assert_almost_equal(result.values, output.values)    
    
# def test_3_2():
#     sample = pd.read_csv("testfiles/data/testout_1.4.csv")
#     output = pd.read_csv("testfiles/data/testout_3.2.csv")
#     result = mylib.corrNearPSD(sample)
#     npt.assert_almost_equal(result.values, output.values)
    
# def test_3_3():
#     sample = pd.read_csv("testfiles/data/testout_1.3.csv")
#     output = pd.read_csv("testfiles/data/testout_3.3.csv")
#     result = mylib.covHigham(sample)
#     npt.assert_almost_equal(result.values, output.values)    
    
# def test_3_4():
#     sample = pd.read_csv("testfiles/data/testout_1.4.csv")
#     output = pd.read_csv("testfiles/data/testout_3.4.csv")
#     result = mylib.corrHigham(sample)
#     npt.assert_almost_equal(result.values, output.values)
    
# def test_4_1():
#     sample = pd.read_csv("testfiles/data/testout_3.1.csv")
#     output = pd.read_csv("testfiles/data/testout_4.1.csv")
#     result = mylib.cholPSD(sample)
#     npt.assert_almost_equal(result.values, output.values)
    
# def test_5_1():
#     sample = pd.read_csv("testfiles/data/test5_1.csv")
#     output = pd.read_csv("testfiles/data/testout_5.1.csv")
#     result = mylib.normalSimulation(np.zeros(5), sample, 100000).cov()
#     npt.assert_almost_equal(result.values, output.values, decimal = 3)
    
# def test_5_2():
#     sample = pd.read_csv("testfiles/data/test5_2.csv")
#     output = pd.read_csv("testfiles/data/testout_5.2.csv")
#     result = mylib.normalSimulation(np.zeros(5), sample, 100000).cov()
#     npt.assert_almost_equal(result.values, output.values, decimal = 3)
    
# def test_5_3():
#     sample = pd.read_csv("testfiles/data/test5_3.csv")
#     output = pd.read_csv("testfiles/data/testout_5.3.csv")
#     result = mylib.normalSimulation(np.zeros(5), sample, 100000).cov()
#     npt.assert_almost_equal(result.values, output.values, decimal = 3)
    
# def test_5_4():
#     sample = pd.read_csv("testfiles/data/test5_3.csv")
#     output = pd.read_csv("testfiles/data/testout_5.4.csv")
#     result = mylib.normalSimulation(np.zeros(5), sample, 100000, fix="higham").cov()
#     npt.assert_almost_equal(result.values, output.values, decimal = 3)
    
# def test_5_5():
#     sample = pd.read_csv("testfiles/data/test5_2.csv")
#     output = pd.read_csv("testfiles/data/testout_5.5.csv")
#     result = mylib.pcaSimulation(np.zeros(5), sample, 100000, .99).cov()
#     npt.assert_almost_equal(result.values, output.values, decimal = 3)
    
def test_6_1():
    sample = pd.read_csv("testfiles/data/test6.csv")
    output = pd.read_csv("testfiles/data/testout6_1.csv")
    result = mylib.arithmetricReturn(sample)
    pdt.assert_frame_equal(result, output, check_index_type=False, check_column_type=False)
    
def test_6_2():
    sample = pd.read_csv("testfiles/data/test6.csv")
    output = pd.read_csv("testfiles/data/testout6_2.csv")
    result = mylib.logReturn(sample)
    pdt.assert_frame_equal(result, output)
    
def test_7_1():
    sample = pd.read_csv("testfiles/data/test7_1.csv")
    output = pd.read_csv("testfiles/data/testout7_1.csv")
    model = mylib.fitNormal(sample)
    result = pd.DataFrame({'mu': model[0], 'sigma': model[1]})
    result.reset_index(drop=True, inplace=True)
    pdt.assert_frame_equal(result, output)

def test_7_2():
    sample = pd.read_csv("testfiles/data/test7_2.csv")
    output = pd.read_csv("testfiles/data/testout7_2.csv")
    model = mylib.fitT(sample)
    result = pd.DataFrame({'mu': [model[1]], 'sigma': [model[2]], 'nu': [model[0]]})
    pdt.assert_frame_equal(result, output)
    
def test_7_3():
    sample = pd.read_csv("testfiles/data/test7_3.csv")
    output = pd.read_csv("testfiles/data/testout7_3.csv")
    model = mylib.treg(sample)
    xn_columns = {f'B{i+1}': [val] for i, val in enumerate(model[1:-3])}
    result = pd.DataFrame({'mu': [model[-1]], 'sigma': [model[-2]], 'nu': [model[-3]], 'Alpha': [model[0]], **xn_columns})
    pdt.assert_frame_equal(result, output)
    
def test_8_1():
    sample = pd.read_csv("testfiles/data/test7_1.csv")
    output = pd.read_csv("testfiles/data/testout8_1.csv")
    var = mylib.var(sample, dist="norm")
    diff = sample.mean() + var
    result = pd.DataFrame({'VaR Absolute': var, 'VaR Diff from Mean': diff})
    result.reset_index(drop=True, inplace=True)
    pdt.assert_frame_equal(result, output)

def test_8_2():
    sample = pd.read_csv("testfiles/data/test7_2.csv")
    output = pd.read_csv("testfiles/data/testout8_2.csv")
    var = mylib.var(sample, dist="t")
    diff = sample.mean() + var
    result = pd.DataFrame({'VaR Absolute': var, 'VaR Diff from Mean': diff})
    result.reset_index(drop=True, inplace=True)
    pdt.assert_frame_equal(result, output, atol=1e-4, check_index_type=False, check_column_type=False)
    
# def test_8_3():
#     sample = pd.read_csv("testfiles/data/test7_2.csv")
#     output = pd.read_csv("testfiles/data/testout8_3.csv")
#     var = mylib.var(sample, dist="sim")
#     diff = sample.mean() + var
#     result = pd.DataFrame({'VaR Absolute': var, 'VaR Diff from Mean': diff})
#     result.reset_index(drop=True, inplace=True)
#     pdt.assert_frame_equal(result, output)

def test_8_4():
    sample = pd.read_csv("testfiles/data/test7_1.csv")
    output = pd.read_csv("testfiles/data/testout8_4.csv")
    es = mylib.es(sample, dist="norm")
    diff = sample.mean() + es
    result = pd.DataFrame({'ES Absolute': es, 'ES Diff from Mean': diff})
    result.reset_index(drop=True, inplace=True)
    pdt.assert_frame_equal(result, output)
    
def test_8_5():
    sample = pd.read_csv("testfiles/data/test7_2.csv")
    output = pd.read_csv("testfiles/data/testout8_5.csv")
    es = mylib.es(sample, dist="t")
    diff = sample.mean() + es
    result = pd.DataFrame({'ES Absolute': es, 'ES Diff from Mean': diff})
    result.reset_index(drop=True, inplace=True)
    pdt.assert_frame_equal(result, output, atol=1e-4)
    
def test_8_6():
    sample = pd.read_csv("testfiles/data/test7_2.csv")
    output = pd.read_csv("testfiles/data/testout8_6.csv")
    es = mylib.es(sample, dist="sim")
    diff = sample.mean() + es
    result = pd.DataFrame({'ES Absolute': es, 'ES Diff from Mean': diff})
    result.reset_index(drop=True, inplace=True)
    pdt.assert_frame_equal(result, output)

def test_9_1():
    pf = pd.read_csv("testfiles/data/test9_1_portfolio.csv")
    ret = pd.read_csv("testfiles/data/test9_1_returns.csv")
    pf.rename(columns={"Stock": "stock", "Holding": "holding", "Starting Price": "price", "Distribution": "dist"}, inplace=True)
    result = mylib.varesSimCopula(pf, ret).values
    output = pd.read_csv("testfiles/data/testout9_1.csv", index_col="Stock").values
    npt.assert_almost_equal(result, output, decimal=3)