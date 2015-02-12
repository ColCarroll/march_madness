import csv
import os
import glob
from contextlib import contextmanager
import sqlite3
from collections import namedtuple

DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(DIR, 'data')


class DataHandler:
    db_name = os.path.join(DATA_DIR, 'kaggle.db')
    data_dir = os.path.join(DIR, 'data')

    def __init__(self):
        pass

    @contextmanager
    def connector(self, commit=True):
        conn = sqlite3.connect(self.db_name)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        try:
            yield cur
        finally:
            if commit:
                conn.commit()
            cur.close()
            conn.close()

    def table_exists(self, table):
        with self.connector(commit=True) as cur:
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name = ?", (table,))
            result = cur.fetchone()
        return result is not None

    def build_table(self, csv_file):
        table_name = os.path.basename(csv_file).split(".")[-2]
        if self.table_exists(table_name):
            return
        column_tuple = self.get_column_tuple(csv_file)
        column_types = ",".join(["{:s} {:s}".format(*column) for column in column_tuple])
        query = "CREATE TABLE %s (%s)" % (table_name, column_types)
        with self.connector(commit=True) as cur:
            cur.execute(query)

        query = "INSERT INTO %s VALUES (%s)" % (table_name, ",".join(["?" for _ in column_tuple]))

        with open(csv_file) as buff:
            with self.connector(commit=True) as cur:
                reader = csv.DictReader(buff)
                for row in reader:
                    cur.execute(query, [row[field[0]] for field in column_tuple])

    def drop_table(self, csv_file):
        table_name = os.path.basename(csv_file).split(".")[-2]
        if not self.table_exists(table_name):
            return
        query = "DROP TABLE %s" % table_name
        with self.connector(commit=True) as cur:
            cur.execute(query)

    def build(self):
        for csv_name in self.list_files():
            self.build_table(csv_name)

    def clean(self):
        for csv_name in self.list_files():
            self.drop_table(csv_name)

    @staticmethod
    def get_column_tuple(csv_file):
        tester = namedtuple('Tester', ["test", "next"])
        testers = {
            "INTEGER": tester(int, "REAL"),
            "REAL": tester(float, "TEXT"),
            "TEXT": tester(lambda _: True, "TEXT"),
        }
        with open(csv_file) as buff:
            reader = csv.DictReader(buff)
            column_types = {field: "INTEGER" for field in reader.fieldnames}
            for row in reader:
                for key, value in row.iteritems():
                    has_type = False
                    while not has_type:
                        try:
                            testers[column_types[key]].test(value)
                            has_type = True
                        except ValueError:
                            column_types[key] = testers[column_types[key]].next
        return column_types.items()

    def list_files(self):
        return glob.glob(os.path.join(self.data_dir, "*.csv"))


def main():
    data = DataHandler()
    data.build()


if __name__ == '__main__':
    main()