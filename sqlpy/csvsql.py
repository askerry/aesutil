import pandas as pd
import sys
sys.path.append('/users/amyskerry/documents/projects/')
from credentials import sqlcfg
from sqlalchemy import create_engine
import os


def create_table_string(path, con, tablename, create_table=False, delimiter=','):
    if create_table:
        temp_name=tablename
    else:
        temp_name='temp'
    #load in a reasonable size chunk of data to give pandas a good chance of getting datatypes rght
    df = pd.read_csv(path, nrows=500000, sep=delimiter)
    #date_cols = [col for col in df.columns if 'date' in col]
    #for col in date_cols:
    #    df[col] = df[col].apply(pd.to_datetime)
    #create empty table from header (with appropriate datatypes)
    df[df.index<0].to_sql(temp_name, con, if_exists='replace', chunksize=1, index=False)
    create_string = con.execute('SHOW CREATE TABLE %s' %temp_name).fetchall()[0][1]
    if not create_table:
        con.execute('DROP TABLE %s' %temp_name)
    create_string = create_string.replace('TABLE %s' %temp_name, 'TABLE %s' %'tablename')
    return create_string


def convert_csv_2_sql(path, con, tablename, delimiter=','):
    path =  os.path.join(os.getcwd(), path)
    create_string = create_table_string(path, con, tablename, create_table=True, delimiter=delimiter)
    print create_string
    con.execute("LOAD DATA INFILE '%s' INTO TABLE %s FIELDS TERMINATED BY '%s' IGNORE 1 LINES;" %(path,tablename, delimiter))


def connect(dbname):
   return create_engine('mysql://%s:%s@%s/%s?charset=utf8' %('root', sqlcfg.passwd, 'localhost', dbname))
    

if __name__=="__main__":
    csvfile, dbname, tablename = sys.argv[1], sys.argv[2], sys.argv[3]
    con = connect(dbname)
    create_string=create_table_string(csvfile, con, tablename, create_table=False)

