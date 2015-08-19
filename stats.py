# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 18:39:21 2015

@author: amyskerry
"""

import numpy as np
import scipy.stats
import statsmodels.api as sma
import statsmodels.formula.api as smfa
import pandas as pd
import pandas.rpy.common as com
base = com.importr('base')
stats = com.importr('stats')
from rpy2.robjects import Formula
from statsmodels.stats.anova import anova_lm



# general modeling functions 

def dummify(df, col, prefix=None, baseline=None):
    '''take dataframe and a categorical column and make dummy variables. optionally specify prefix and baseline level
    output: new df, new dummy columns'''
    if prefix is None:
        prefix=col
    dummies = pd.get_dummies(df[col], prefix=prefix)
    if baseline is None:
        baseline=dummies.columns[0]
    else:
        baseline=prefix+'_'+baseline
    cols=[c for c in dummies.columns if c!=baseline]
    print "DUMMY CODING: {} level {} set to baseline, compared to {}".format(col, baseline, ', '.join(cols))
    print "***************************************************************************************"
    df=df.join(dummies.ix[:,cols])
    del df[col]
    return df, cols
    
def addintercept(df):
    '''manually add intercept to the dataframe'''
    df['Intercept']=[1 for i in range(len(df))]
    return df
    
def zscorecol(df, col):
    return (df[col] - df[col].mean())/df[col].std(ddof=0)
    
def bincolumn(df, col, bins=5):
    binned=scipy.stats.binned_statistic(df[col], df[col], bins=bins)
    df['{}_binned'.format(col)]=binned[2]
    df['{}_binned'.format(col)]=df['{}_binned'.format(col)].apply(lambda x: "{} - {}".format(binned[1][x], binned[1][x-1]))
    return df
    

# regression wrapper (sfma.logit, poisson, probit, ols)

def regressionwrapper(df, ycol, xcols=None, interactionpairs=None, modeltype='linear', normalize=True):
    '''perform logistic regression predicting ycol as a function of features in xcols, and feature pairs in interaction pairs
    output: df, result '''
    ldf, xcols, interactionpairs = prepdf(df, ycol, xcols, interactionpairs, normalize=True)
    result=fitmodel(ldf, ycol, xcols, modeltype, interactionpairs)
    return ldf, result
    
def prepdf(df, ycol, xcols, interactionpairs=None, normalize=True):
    '''prepare dataframe for logistic regression'''
    if interactionpairs is None:
        interactionpairs=[]
    if xcols is None:
        xcols=list(df.columns)
        xcols.remove(ycol)
    ldf=df[[ycol]+xcols]
    catvars=[c for c in ldf.columns if ldf[c].dtype==np.dtype('O')]
    for c in catvars:
        if len(ldf[c].unique())==2:
            cat1=ldf[c].unique()[0]
            cat2=ldf[c].unique()[1]
            ldf[c]=ldf[c].replace({cat1:0, cat2:1})
            print "recoding column {}: {} as 0, {} as 1".format(c, cat1, cat2)
        else:
            ldf, newcols=dummify(ldf, c)
            xcols.remove(c)
            xcols.extend(newcols)
            for intpair in interactionpairs:
                if c in intpair:
                    pairother = [i for i in intpair if i != c][0]
                    newpairs=[[pairother, newother] for newother in newcols]
                    interactionpairs.remove(intpair)
                    interactionpairs.extend(newpairs)
    ldf=addintercept(ldf)
    for col in xcols:
        ldf[col]=zscorecol(ldf, col)
    return ldf, xcols, interactionpairs

def fitmodel(ldf, ycol, xcols, modeltype, interactionpairs):
    string= "{} ~ {}".format(ycol, ' + '.join(xcols))
    for intpair in interactionpairs:
        string += ' + '+intpair[0]+':'+intpair[1]
    print "Running {} regression model:".format(modeltype)
    print string
    print "***************************************************************************************"
    print "***************************************************************************************"
    if modeltype=='logistic':
        model = smfa.logit(string, ldf)
    elif modeltype=='linear':
        model = smfa.ols(string, ldf)
    elif modeltype=='poisson':
        model = smfa.poisson(string, ldf)
    elif modeltype=='probit':
        model = smfa.probit(string, ldf)
    result=model.fit(maxiter=10000)
    return result
    
def apireminder(modeltype='linear'):
    print "Review of relevant functions for interpreting/summarizing {} regression".formt(modeltype)
    print "***************************************************************************************"
    print "  - For all model types, can use result.summary() to get summary of the fitted model"
    print "  - Use result.params to get coefficients and result.conf_int() to get confidence intervals on coefficients"
    if modeltype=='linear':
        print "  - import anova_lm from statsmodels.stats.anova and run lm_anova(result) to get anova table"
    if modeltype=='logistic':
        print "  - Use result.pred_table() to get predictions"
        print "  - Remember that coefficients of a logistic regression are NOT the rate of change in Y for every unit in X (as in OLS)... instead coefficient is interpreted as the rate of change in the 'log odds' as X changes (not very intuitive)."
        print "     - Instead compute the more intuitive 'marginal effect' of IV on the probability. The marginal effect is dp/dB = f(BX)B. The marginal effects depend on the values of the IVs, but we often evaluate the marginal effects at the means of the IVs."
        print "     - To get the marginal effects: mfx = result.get_margeff() .... then call mfx.summary()"

# simple anova wrappers

def twowayanova(y,factor1, factor2, df, repeatedmeasures=False, withinunit=None):
    if repeatedmeasures:
        df=droppairsmissingdata(df, factor1, withinunit)
        df=droppairsmissingdata(df, factor2, withinunit)
        r=twowayanova_within_R(y,factor1,factor2,withinunit,df)
    else:
        r=twowayanova_between(y,factor1,factor2,df)
    return r

def twowayanova_within_R(y, factor1, factor2, withinunit, df):
    '''currently only supports 2 factors. R style, since statsmodel can't do repeated measures yet. 
    treats withinunit as the index for repeated measurements'''
    fml = '{0} ~ {1} + {2} + {1}:{2} + Error({3}/ ({1} + {2} + {1}:{2}))'.format(y,factor1,factor2,withinunit)  #  formula string. note that you need to explicitly specify main effects an interaction in the error term. check output against output on vassarstats
    print 'two way RM anova: {}'.format(fml)        
    dfr = com.convert_to_r_dataframe(df, True)  # convert from pandas to R and make string columns factors
    fml_ = Formula(fml)  #  make a formula    obect
    result=base.summary(stats.aov(fml_, dfr))
    print result
    return result

def twowayanova_between(y, factor1, factor2, df):
    '''two way anova'''
    fml = '{} ~ {} * {}'.format(y, factor1, factor2)#shorthand for feature + roi + feature:roi
    print 'two way anova: {}'.format(fml)        
    lm=ols(fml, df).fit()
    #print lm_.summary()
    print anova_lm(lm)
    return anova_lm(lm)
    
def droppairsmissingdata(df, factor, paircol):
    values=df[factor].unique()
    pairs=df[paircol].unique()
    initialpairs=len(pairs)
    for v in values:
        thisset=df[df[factor]==v][paircol].unique()
        pairs=[s for s in pairs if s in thisset]
    df=df[[row[paircol] in pairs for index,row in df.iterrows()]]
    if len(pairs)<initialpairs:
        print "reduced from {} to {} units".format(initialpairs, len(pairs))
    return df

# misc logistic functions
    
def inverselogit(row, ycol, params=None):
    ''' get probability from feature vector'''
    val = params[0] #intercept
    for r in row.keys():
        if r !=ycol:
            val += row.loc[r]*params.loc[r]
    return 1/(1+np.exp(-val))
    
def prob(result, inputdict):
    '''takes a statsmodels result object and an input dictionary of values (corresponding to coefficient names in results object) and returns probability'''
    ymxb=result.params['Intercept']
    for key in inputdict:
        ymxb+=result.params[key]*inputdict[key]
    return 1/(1 + np.exp**(-1*ymxb))

    
# basic statistics utilities

def nancorr(x,y):
    '''takes two vectors and reduces to vectors corresponding only to indices where both vectors are nonnan. returns correlation'''
    zipped=zip(x,y)
    nonnan=[tup for tup in zipped if not any(np.isnan(tup))]
    x,y=zip(*nonnan)
    if len(y) != len(x) or len(y)<2:
        print "warning: vector lengths don't make sense"
    rvalue, pvalue=scipy.stats.pearsonr(x,y)
    return rvalue, pvalue, len(x)
    
def diffcorrcoeftest(rvalue1, rvalue2, N1, N2):
    ''' tests for difference between 2 r values by performing fishers transformation and performing t-test on the z values. returns z and two tailed p '''
    r_z1=np.arctanh(rvalue1) #equivalent to 0.5 * np.log((1 + rvalue1)/(1 - rvalue1))
    r_z2=np.arctanh(rvalue2)
    se_diff_r = np.sqrt(1.0/(N1 - 3.0) + 1.0/(N2 - 3.0))
    diff = r_z1 - r_z2
    z = abs(diff / se_diff_r)
    p = (1 - scipy.stats.norm.cdf(z))*2
    return z,p
    
def diff2proportions(prop1,n1,prop2,n2):
    '''takes two proportions and their Ns and returns the z and p'''
    p = (prop1*n1 + prop2*n2) / (n1 + n2)
    SE = np.sqrt(p*(1-p) *  ((1.0/n1) + (1.0/n2)))
    z = (np.abs(prop1 - prop2)) / SE
    pval = scipy.stats.norm.sf(z)*2
    return z, pval
    
def chisquare_independence(mat):
    '''performs chi-square test assessing whether row variable is independent of column variable'''
    chi2,p,dof, expected = scipy.stats.chi2_contingency(mat)
    print "chi-squared(df={})={:.2f}, p={:.3f}".format(dof, chi2, p)
    return chi2, p, dof, expected
    
### misc bootstrap CI functions    

def bootstrap(data, num_samples=500, samplesize=None, statistic=np.mean, alpha=.05,plotit=False):
    """Returns bootstrap estimate of 100.0*(1-alpha) CI for statistic."""
    n = len(data)
    data=np.array(data)
    if samplesize==None:
        samplesize=n
    idx=bootstrapidx(n, num_samples, samplesize)
    samples = data[idx]
    stat = np.sort(statistic(samples, 1))
    observedmean=np.mean(data)
    lowerbound, upperbound, SEM = sampledist(stat, num_samples, alpha, plotit=False)
    nullrejected, p=testdist(observedmean, stat, num_samples, alpha)
    return observedmean, lowerbound, upperbound, SEM
    
def bootstrapidx(fulldata_length, num_samples, samplesize):
    '''returns set of indices (num_samples x samplesize for sampling from the fulldata'''
    idx = np.random.randint(0, fulldata_length, (num_samples, samplesize))
    return idx
    
def sampledist(samplestatistics, num_samples, alpha, plotit=True, observed=None):
    '''takes set of sample statistics and returns CI and SEM'''
    lowerbound=np.sort(samplestatistics)[int((alpha/2.0)*num_samples)]
    upperbound=np.sort(samplestatistics)[int((1-alpha/2.0)*num_samples)]
    SEM=np.std(samplestatistics,ddof=1)
    return lowerbound, upperbound, SEM
    
def testdist(observedmean, samplemeans, num_samples, alpha, tail='both'):
    '''takes set of samples, an observation, and an alpha, and returns whether null hypothesis is rejected at that alpha'''
    pdict={0:'>{}'.format(alpha),1:'<{}'.format(alpha)}
    lowerbound, upperbound, SEM = sampledist(samplemeans, num_samples, alpha, observed=observedmean)
    if tail=='both':
        h=observedmean<lowerbound or observedmean>upperbound
    elif tail=='right':
        h=observedmean>upperbound
    elif tail=='left':
        h=observedmean<lowerbound
    pstr=pdict[h]
    return h,pstr
    
def kappa(y_true, y_pred, weights=None, allow_off_by_one=False):
    """
    Calculates the kappa inter-rater agreement between two the gold standard
    and the predicted ratings. Potential values range from -1 (representing
    complete disagreement) to 1 (representing complete agreement).  A kappa
    value of 0 is expected if all agreement is due to chance.

    In the course of calculating kappa, all items in `y_true` and `y_pred` will
    first be converted to floats and then rounded to integers.

    It is assumed that y_true and y_pred contain the complete range of possible
    ratings.

    This function contains a combination of code from yorchopolis's kappa-stats
    and Ben Hamner's Metrics projects on Github.

    :param y_true: The true/actual/gold labels for the data.
    :type y_true: array-like of float
    :param y_pred: The predicted/observed labels for the data.
    :type y_pred: array-like of float
    :param weights: Specifies the weight matrix for the calculation.
                    Options are:

                        -  None = unweighted-kappa
                        -  'quadratic' = quadratic-weighted kappa
                        -  'linear' = linear-weighted kappa
                        -  two-dimensional numpy array = a custom matrix of
                           weights. Each weight corresponds to the
                           :math:`w_{ij}` values in the wikipedia description
                           of how to calculate weighted Cohen's kappa.

    :type weights: str or numpy array
    :param allow_off_by_one: If true, ratings that are off by one are counted as
                             equal, and all other differences are reduced by
                             one. For example, 1 and 2 will be considered to be
                             equal, whereas 1 and 3 will have a difference of 1
                             for when building the weights matrix.
    :type allow_off_by_one: bool
    """
    from sklearn.metrics import confusion_matrix
    from six import string_types

    # Ensure that the lists are both the same length
    assert(len(y_true) == len(y_pred))

    # This rather crazy looking typecast is intended to work as follows:
    # If an input is an int, the operations will have no effect.
    # If it is a float, it will be rounded and then converted to an int
    # because the ml_metrics package requires ints.
    # If it is a str like "1", then it will be converted to a (rounded) int.
    # If it is a str that can't be typecast, then the user is
    # given a hopefully useful error message.
    # Note: numpy and python 3.3 use bankers' rounding.
    try:
        y_true = [int(np.round(float(y))) for y in y_true]
        y_pred = [int(np.round(float(y))) for y in y_pred]
    except ValueError as e:
        raise e

    # Figure out normalized expected values
    min_rating = min(min(y_true), min(y_pred))
    max_rating = max(max(y_true), max(y_pred))

    # shift the values so that the lowest value is 0
    # (to support scales that include negative values)
    y_true = [y - min_rating for y in y_true]
    y_pred = [y - min_rating for y in y_pred]

    # Build the observed/confusion matrix
    num_ratings = max_rating - min_rating + 1
    observed = confusion_matrix(y_true, y_pred,
                                labels=list(range(num_ratings)))
    num_scored_items = float(len(y_true))

    # Build weight array if weren't passed one
    if isinstance(weights, string_types):
        wt_scheme = weights
        weights = None
    else:
        wt_scheme = ''
    if weights is None:
        weights = np.empty((num_ratings, num_ratings))
        for i in range(num_ratings):
            for j in range(num_ratings):
                diff = abs(i - j)
                if allow_off_by_one and diff:
                    diff -= 1
                if wt_scheme == 'linear':
                    weights[i, j] = diff
                elif wt_scheme == 'quadratic':
                    weights[i, j] = diff ** 2
                elif not wt_scheme:  # unweighted
                    weights[i, j] = bool(diff)
                else:
                    raise ValueError('Invalid weight scheme specified for '
                                     'kappa: {}'.format(wt_scheme))

    hist_true = np.bincount(y_true, minlength=num_ratings)
    hist_true = hist_true[: num_ratings] / num_scored_items
    hist_pred = np.bincount(y_pred, minlength=num_ratings)
    hist_pred = hist_pred[: num_ratings] / num_scored_items
    expected = np.outer(hist_true, hist_pred)

    # Normalize observed array
    observed = observed / num_scored_items

    # If all weights are zero, that means no disagreements matter.
    k = 1.0
    if np.count_nonzero(weights):
        k -= (sum(sum(weights * observed)) / sum(sum(weights * expected)))

    return k