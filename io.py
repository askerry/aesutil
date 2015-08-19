# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 18:03:47 2014

@author: amyskerry
"""

# basic file io
import pickle
import csv

################################
# #####  Misc file io    #######
################################

def pickletheseobjects(filename, objects):
    '''takes list of objects and pickles them to <filename>'''
    with open(filename, 'wb') as output:
        pickler = pickle.Pickler(output, pickle.HIGHEST_PROTOCOL)
        for obj in objects:
            pickler.dump(obj)
            
def loadpickledobjects(filename):
    '''takes <filename> and loads it, returning list of objects'''
    with open(filename, 'r') as inputfile:
        remaining=1
        objects=[]
        while remaining:
            try:
                objects.append(pickle.load(inputfile))
            except:
                remaining=0
    return objects
    
def extractcsvdata(csvfile, headings=True):
    '''takes <csvfile> and returns colnames and data (if headings==True), or just data (if headings==False)'''
    with open(csvfile, 'rU') as csvf:
        reader = csv.reader(csvf)
        if headings:
            colnames=reader.next()
        data=[row for row in reader]
    if headings:
        return colnames, data
    else:
        return data
    
def writecsvdata(filename,data, colnames=None):
    '''take iterable data (and optional colnames) and save to csvfile <filename>'''
    with open(filename, 'w') as csvf:
        writer = csv.writer(csvf)
        if colnames:
            writer.writerow(colnames)
        for row in data:
            writer.writerow(row)
            
def extracttxtdata(txtfile):
    with open(txtfile, 'rU') as f:
        lines=[line.rstrip('\n') for line in f]
    return lines

def writetxtdata(filename, lines):
    with open(filename, 'w') as f:
        for line in lines:
            line='{}\n'.format(line)
            f.write(line)
