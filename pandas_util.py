import pandas as pd
import sklearn.feature_extraction as sfk
import numpy as np
import date_helpers as dh


table = {'week': 'W', 'weeks': 'W', 'days': 'D', 'day': 'D', 'month': 'M', 'months': 'M', 'year': 'Y', 'years': 'Y', 'hours': 'h',
         'hour': 'h', 'hr': 'h', 'minute': 'm', 'minutes': 'm', 'min': 'm', 'seconds': 's', 'second': 's', 'sec': 's', 'ms': 'ms', 'ns': 'ns'}


################################
# #### df level operations #####
################################

def get_df_dtypes(df):
    '''print columns for each datatype'''
    dtypedf = pd.DataFrame(columns=['dtype'], data=df.dtypes)
    dtypedf['count'] = 1
    grouped = dtypedf.groupby('dtype')
    print grouped.count()
    for g, data in grouped:
        print "data type %s:" % g
        print "   " + ', '.join(data.index.values)
        print " "
    return dtypedf


def get_count_df(timecol, conditioncol, df):
    '''counts occurences of unit in timecol, grouped by dimension in condition col'''
    gdf = df.groupby([timecol, conditioncol]).count()
    countcol = gdf.columns[0]
    gdf = gdf[[countcol]].reset_index()
    gdf = gdf.rename(columns={countcol: 'count'})
    return gdf


def expand_timeseries_df(df, on_col, unit=days):
    '''take a df and a time-based column and expand column to full series (fillingin with np.nan)'''
    newvalues = pu.make_tseries(df[on_col], unit=unit)
    extended = pd.DataFrame(data=newvalues, columns=[on_col])
    return pd.merge(extended, df, on=on, how='left')


def expand_groupby(df, grouping_cols, count_cols=None, fill_cols=None):
    '''perform an expanded groupby with option to specify all values to exist in each groupby combination (default to product of all values in all grouping cols)
    INPUTS:
        df (pandas dataframe): initial data frame
        grouping_cols (list or dictionary): specifies grouping columns and possible values for each
            if possible values for a column are None, this will be set to all values in the column
            if specified as a list, this will create a dictionary with all possible values for each
        count_cols (list): columns for which we want the aggregation function to be a count
        fill_cols (list): columns for which we want the aggregation function to be a mode
    OUPUTS:
        expanded_df (pandas dataframe): expanded multiindex dataframe where each every possible combination of groupings is represented
    '''
    if not isinstance(grouping_cols, dict):
        grouping_cols = {col: None for col in grouping_cols}
    # for count columns, specify count (len) as aggregration operation
    aggdict = {col: lambda x: len(x) for col in count_cols}
    # for fill columns we will simple fill in with modal value
    for c in fill_cols:
        if c not in aggdict:
            aggdict[c] = lambda x: scipy.stats.mode(x)[0][0]
    col_names = grouping_cols.keys()
    col_values = []
    for col in col_names:
        if grouping_cols[col] is None:
            col_values.append(df[col].unique())
        else:
            col_values.append(grouping_cols[col])
    # create full set of possible grouping col combinations and join with
    # dataframe
    full_indexdf = pd.DataFrame(
        index=pd.MultiIndex.from_product(col_values, names=col_names))
    grouped = df.groupby(col_names).agg(aggdict)
    return full_indexdf.join(grouped)


################################
# ## column level operations ###
################################

def make_tseries_from_tstampdf(timestamp_df, timestamp_col, value_cols=None, unit='days'):
    '''take a dataframe with a timestamp col and return a new df with full timeseries (by default, filled in values left as nan'''
    if value_cols is None:
        value_cols = [
            col for col in timestamp_df.columns if col != timestamp_col]
    unitstr = table[unit]
    ts_list = []
    for col in value_cols:
        ts = pd.Series(
            timestamp_df[col].values, timestamp_df[timestamp_col].values)
        ts_list.append(ts.asfreq('1' + unitstr, method=None))
    return pd.concat(ts_list, axis=1)


def make_tseries(timestamp_series, unit='days'):
    '''take a series of timestamps and return a timeseries with frequency of unit'''
    unitstr = table[unit]
    ts = pd.Series([1, timestamp_series.values)
    return ts.asfreq('1' + unitstr, method=None).index.values


def round_timestamp(timestamp_series, unit='days'):
    '''round timestamp to appropriate time unit'''
    unitstr = table[unit]
    try:
        timestamp_series = pd.to_datetime(timestamp_series)
        timestamp_series = timestamp_series.apply(
            lambda x: np.datetime64(x, unitstr))
    except:
        timestamp_series_str = timestamp_series.apply(
            dh.round_date, unit=unitstr)
        if unitstr == 'Y':
            return timestamp_series_str.astype(int)
        else:
            timestamp_series = pd.to_datetime(timestamp_series_str)
    return timestamp_series.values


def compute_timedelta(timestamp_series, unit='days', startdate=None):
    '''take a series and return delta from startdate (default = min of series) in unit days'''
    unit
    timestamp_series = pd.to_datetime(timestamp_series)
    if startdate is None:
        startdate = timestamp_series.min()
    timeseries_delta = timestamp_series - startdate
    return timeseries_delta.astype('timedelta64[%s]' % table[unit]).values


def compute_cyclical_time(timestamp_series, unit='dayofweek'):
    '''take a series and return some cyclical time unit'''
    # unit: 'second', 'minute', 'hour', 'day', 'dayofweek', 'week', 'month',
    # 'year'
    timestamp_series = pd.DatetimeIndex(timestamp_series)
    return getattr(timestamp_series, unit)


def vectorize_text_col(dftextseries, vectorizer=None):
    '''take a text-based series and return dataframe with vectorized representation'''
    if vectorizer is None:
        vectorizer = skf.text.CountVectorizer()
    counts = vectorizer.fit_transform(dftextseries.values)
    terms = vectorizer.get_feature_names()
    return pd.DataFrame(columns=terms, data=counts, index=dftextseries.index.values)


################################
# #### row level operations ####
################################

def encode_ascii(string):
    return string.strip().encode('ascii', 'ignore')


def encode_utf8(string):
    return string.strip().encode('utf-8', 'ignore')


def fix_timestamp(timestamp):
    if len(timestamp) == 13 and '.' not in timestamp:
        timestamp = float(timestamp) / 1000
    return timestamp


##########################################################################
# #### Misc Cookbooks (reminder of handy builtin functionality I otherwise forget about)  ##
##########################################################################

# ##### Can use all standard string operations ######

def contains(df, column, substring):
    '''return boolean specifying whether column contains substring'''
    return df[columun].str.contains(substring)


def upper(df, column):
    '''return column all uppercase'''
    return df[column].str.upper()


def lower(df, column):
    '''return column all lowercase'''
    return df[column].str.lower()


def count_substring_occurence(df, column, substring):
    ''''return count of substring in column value'''
    return df[column].str.count(substring)


def endswith(df, column, substring):
    '''return boolean specifying whether column value ends with substring'''
    return df[column].str.endswith(substring)


def startswith(df, column, substring):
    '''return boolean specifying whether column value starts with substring'''
    return df[column].str.startswith(substring)


def split(df, column, splitter=' '):
    '''return series of lists containing string split by splitter'''
    return df[column].str.split(splitter)


def strip(df, column):
    '''return column with whitespace stripped'''
    return df[column].str.strip()


def match_regex(df, column, pattern, case=True, flags=0):
    '''return boolean specifying whether regex pattern matches'''
    return df[column].str.match(pattern, case=case, flags=flags, as_indexer=True)


def extract_regex(df, column, pattern, case=True, flags=0):
    '''return regex matches with nan for nonmatches'''
    return df[column].str.match(pattern, case=case, flags=flags, as_indexer=True)

def replace(df, column, pattern, newstr, case=True):
    '''replace substring or regex pattern with newstr'''
    return df[column].str.replace(pattern, newstr, case=case)

def easy_vectorize(df, column, sep=' '):
    '''split by sep and create bag of words (1 or 0) vectorized representation of the string'''
    return df[column].str.get_dummies(sep=sep)


#### numeric operations ########

def absolute_value(df, column):
    '''returns absolute value of column as series'''
    return df[column].abs()


def argmax(df, column):
    '''returns index of min of column'''
    return df[column].argmax()


def argmin(df, column):
    '''returns index of min of column'''
    return df[column].argmin()


def argsort(df, column):
    '''return indices sorted by values in column'''
    return df[column].argsort().index


def between(df, column, low, high):
    '''returns boolean for whether value of column is between low and high'''
    return df[column].between(low, high)


def corr(df, col1, col2, method='pearson'):
    '''returns corrleation between two columns (w/ nan handling). defaults to pearson'''
    return df[col1].corr(col2, method=method)


def covariance(df, col1, col2):
    '''returns covariance between two columns'''
    return df[col1].cov(col2

def dot_product(df, col1, col2):
    '''returns dot product of two columns'''
    return df[col1].dot(col2


def interpolate(df, column, method='linear'):
    ''' interpolate missing values in column
    method : {'linear', 'time', 'index', 'values', 'nearest', 'zero',
          'slinear', 'quadratic', 'cubic', 'barycentric', 'krogh',
          'polynomial', 'spline' 'piecewise_polynomial', 'pchip'}

    * 'linear': ignore the index and treat the values as equally
      spaced. default
    * 'time': interpolation works on daily and higher resolution
      data to interpolate given length of interval
    * 'index', 'values': use the actual numerical values of the index
    * 'nearest', 'zero', 'slinear', 'quadratic', 'cubic',
      'barycentric', 'polynomial' is passed to
      `scipy.interpolate.interp1d` with the order given both
      'polynomial' and 'spline' requre that you also specify and order
      (int) e.g. df.interpolate(method='polynomial', order=4)
    * 'krogh', 'piecewise_polynomial', 'spline', and 'pchip' are all
      wrappers around the scipy interpolation methods of similar
      names. See the scipy documentation for more on their behavior:
      http://docs.scipy.org/doc/scipy/reference/interpolate.html#univariate-interpolation
      http://docs.scipy.org/doc/scipy/reference/tutorial/interpolate.html'''
    return df[column].interpolate(method=method)


def percentile(df, column, quantile=.5):
    '''returns value at quantile percentile of the column'''
    return df[column].quantile(quantile)

def kurtosis(df, column):
    return df[column].kurtosis()

def skew(df, column):
    return df[column].skew()

def rank(df, column):
    return df[column].rank()

def sem(df, column):
    return df[column].sem()

def autocorrelation(df, column):
    '''return lag-1 autocorrelation for column'''
    return df[column].autocorr()

