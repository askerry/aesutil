import time
import datetime
import numpy as np
import pandas as pd
import os
import dateutil
from dateutil import zoneinfo


#########################
# #### Conversions ######
#########################


def datetime_to_npdatetime(datetime_obj):
    return np.datetime64(datetime_obj)


def npdatetime_to_datetime(npdatetime_obj):
    pdts = pd.to_datetime(npdatetime_obj)
    return pdts.to_datetime()


def npdatetime_to_string(npdatetime_obj, format='%m/%d/%y'):
    pdts = pd.to_datetime(str(npdatetime_obj))
    return pdts.strftime(format)


def datetime_to_string(datetime_obj, format='%m/%d/%y'):
    return datetime_obj.strftime(format)


def string_to_datetime(string, format='%m/%d/%y'):
    return datetime_obj.strptime(string, format)


def string_to_npdatetime(string, format='%m/%d/%y'):
    dt = datetime_obj.strptime(string, format)
    return np.datetime64(dt)


def string_to_pdtimestamp(string, format=None):
    pd.to_datetime(string, format=format)


def rawtimestamp_to_npdatetime(timestamp):
    pdts = pd.Timestamp.fromtimestamp(timestamp)
    return pdts.to_datetime64()


def rawtimestamp_to_datetime(timestamp):
    return datetime.datetime.utcfromtimestamp(timestamp)


def pdtimestamp_to_datetime(pdtimestamp):
    return pdtimestamp.to_datetime()


def pdtimestamp_to_npdatetime(pdtimestamp):
    return pdtimestamp.to_datetime64()


def datetime_to_pdtimestamp(datetime_obj):
    return pd.Timestamp(datetime_obj)


def npdatetime_to_pdtimestamp(npdatetime_obj):
    return pd.Timestamp(npdatetime_obj)


#############################
# ####### Time Zones ########
#############################

def list_timezones():
    '''list all available timezones'''
    for timezone_path in dateutil.tz.TZPATHS:
        try:
            zones = []
            for i in os.listdir(timezone_path):
                if i != '+VERSION' and '.' not in i:
                    try:
                        for ii in os.listdir(os.path.join(timezone_path, i)):
                            zones.append('%s/%s' % (i, ii))
                    except:
                        zones.append(i)
            return zones
        except:
            pass


def search_timezones(search_str):
    '''search for timezone info by substring'''
    tzones = list_timezones()
    matches = []
    for zone in tzones:
        if search_str.lower() in zone.lower():
            matches.append(zone)
    return matches


def get_tz_info(tzname):
    '''get timezone info based on name'''
    return zoneinfo.gettz(tzname)


def make_time_aware(datetime_obj, tzname='UTC'):
    '''take a naive datetime object and make it timezone aware'''
    tzinfo = zoneinfo.gettz(tzname)
    return datetime_obj.replace(tzinfo=tzinfo)


def convert_timezones(datetime_obj, dest_tzname='UTC'):
    '''take timezone aware datetime and convert to other timezone'''
    return datetime_obj.astimezone(zoneinfo.gettz(dest_tzname))


#############################
# ####### Rounding ##########
#############################

table = {'week': 'W', 'weeks': 'W', 'days': 'd', 'day': 'd', 'D': 'd', 'd': 'd', 'month': 'm', 'months': 'm', 'year': 'y', 'years': 'y', 'Y': 'y', 'y': 'y', 'hours': 'H',
         'hour': 'H', 'hr': 'H', 'minute': 'M', 'minutes': 'M', 'min': 'M', 'seconds': 'S', 'second': 'S', 'sec': 'S', 'ms': 'f'}


def round_date(input_obj, unit='day'):
    unitstr = table[unit]
    if isinstance(input_obj, np.datetime64):
        date_obj = input_obj.to_datetime()
    else:
        date_obj = input_obj
    units = ['Y', 'M', 'D', 'h', 'm',
             's', 'ms', 'ns']
    i = units.index(unitstr)
    datelist = []
    for u in units[:i]:
        datelist.append(getattr(date_obj, u))
    datestr = '-'.join(datelist)
    formatstr = '_'.join(units[:1])
    dt = date_obj.strptime(datestr, formatstr)
    if isinstance(input_obj, np.datetime64):
        return np.datetime64(dt)
    elif isinstance(input_obj, pd.tslib.Timestamp):
        return pd.Timestamp(dt)
    else:
        return dt
