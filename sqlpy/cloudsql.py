import MySQLdb
import os
import sys
sys.path.append('/users/amyskerry/documents/projects')
from credentials import sqlcfg


mapping= {'text':"STRING",'char':"STRING",'varchar':"STRING",'int':"INTEGER", 'tinyint':"INTEGER",'smallint':"INTEGER",'mediumint':"INTEGER",'bigint':"INTEGER",'float':"FLOAT",'double':"FLOAT",'decimal':"FLOAT",'bool':"BOOLEAN",'date':"TIMESTAMP",'datetime':"TIMESTAMP"}


def query(cursor, query):
    cursor.execute(query)
    return cursor.fetchall()

def local_sql_connect(dbname):
    con=MySQLdb.connect('localhost', 'root', sqlcfg.passwd, dbname)
    cursor=con.cursor()
    return con, cursor


def connect_cloudsql(cloudname, cloudip):
    env = os.getenv('SERVER_SOFTWARE')
    if (env and env.startswith('Google App Engine/')):
        # Connecting from App Engine
      db = MySQLdb.connect(
          unix_socket='/cloudsql/%s:sql' %cloudname,
          user='root')
    else:
        # Connecting from an external network.
        # Make sure your network is whitelisted
      db = MySQLdb.connect(
          host=cloudip,
          port=3306,
          user='askerry', passwd=sqlcfg.passwd)

    cursor = db.cursor()
    print query(cursor, 'show databases')
    return cursor

