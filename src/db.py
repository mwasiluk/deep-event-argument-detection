import traceback

import psycopg2
import sys
import json

class DatabaseMng:

    def __init__(self, dbname, user, host, password, table_name):
        self.table_name = table_name
        self.password = password
        self.host = host
        self.user = user
        self.dbname = dbname

    def connect(self):
        try:
            self.conn = psycopg2.connect("dbname='"+self.dbname+"' user='"+self.user+"' host='"+self.host+"' password='"+self.password+"'")
        except:
            print "unable to connect to the database"

    def get_create_table_statement(self, table_name):
        return """CREATE TABLE IF NOT EXISTS public."""+table_name+"""
                (
                  id serial,
                  config jsonb,
                  p_r_f jsonb,
                  acc double precision,
                  model_summary text,
                  model_id text,
                  num_classes integer,
                  class_weights jsonb,
                  labels_index jsonb,
                  CONSTRAINT """+table_name+"""_pkey PRIMARY KEY (id)
                )"""

    def create_table(self):
        create_statement = self.get_create_table_statement(self.table_name)
        try:
            cur = self.conn.cursor()
            cur.execute(create_statement)
            self.conn.commit()
        except:
            print "unable to create table"
            traceback.print_exc()


    def init(self):
        self.connect()
        self.create_table()

    def save_result(self, config, acc, p_r_f, model_summary, num_classes, class_weights, labels_index, model_id):

        prf = {}
        m = ['P', 'R', "F1", "S"]

        for i,a in enumerate(p_r_f):
            prf[m[i]] = {}
            for vi, v in enumerate(a.tolist()):
                prf[m[i]][labels_index.keys()[labels_index.values().index(vi)]] = v

        try:
            cur = self.conn.cursor()

            cur.execute("""INSERT INTO """+self.table_name+"""(config,acc, p_r_f, model_summary, num_classes, class_weights, labels_index, model_id) VALUES (%(config)s, %(acc)s, %(p_r_f)s, %(model_summary)s, %(num_classes)s, %(class_weights)s, %(labels_index)s, %(model_id)s)""",
                        {
                            'config': json.dumps(config),
                            'acc': acc,
                            'p_r_f': json.dumps(prf),
                            'model_summary': model_summary,
                            'model_id': str(model_id),
                            'num_classes': num_classes,
                            'class_weights': json.dumps(class_weights.tolist()),
                            'labels_index': json.dumps(labels_index)
                        })
            self.conn.commit()
        except:
            print "unable to save result"
            traceback.print_exc()

