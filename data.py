import csv
import json
import os
import glob
from contextlib import contextmanager
import sqlite3
from collections import namedtuple
import re

DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(DIR, 'data')
JSON_DIR = os.path.join(DIR, 'model_json')


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


class NameMap:
    fname = os.path.join(DATA_DIR, "team_spellings.csv")

    def __init__(self):
        self._data = None

    @property
    def data(self):
        if self._data is None:
            with open(self.fname) as buff:
                self._data = {row["name_spelling"].lower(): int(row["team_id"]) for row in csv.DictReader(buff)}
        return self._data

    def lookup(self, team_name):
        try:
            return self.data[team_name.lower()]
        except KeyError:
            print(team_name)


class TourneyTeam:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._round_probs = None

    def round_probs(self):
        if self._round_probs is None:
            self._round_probs = [0 for _ in range(7)]
            for key, value in self.kwargs.iteritems():
                match = re.match(r"rd(\d)_win", key)
                if match:
                    round = int(match.group(1))
                    self._round_probs[round - 1] = float(value)
        return self._round_probs

    def __repr__(self):
        return self.kwargs["team_name"]


def id_to_name():
    with DataHandler().connector() as cur:
        cur.execute("SELECT * FROM teams")
        data = {row["team_id"]: row["team_name"] for row in cur}
    return data


def team_seeds():
    with DataHandler().connector() as cur:
        cur.execute("SELECT * FROM tourney_seeds WHERE season=2015")
        data = {row["team"]: int(row["seed"][1:3]) for row in cur}
    return data

def human_readable():
    names = id_to_name()
    seeds = team_seeds()
    data = []
    with open(os.path.join(DIR, 'out_data/predictions/2015_ensemble.csv')) as buff:
        reader = csv.DictReader(buff)
        for row in reader:
            _, team_one, team_two = map(int, row["id"].split("_"))
            pred = 100 * float(row["pred"])
            team_one = "({:d}) {:s}".format(seeds[team_one], names[team_one])
            team_two = "({:d}) {:s}".format(seeds[team_two], names[team_two])
            data.append({"team_one": team_one, "team_two": team_two, "prediction": pred})
    json.dump({"seeds": data}, open(os.path.join(DATA_DIR, "preds.json"), 'w'))



def check_first_round():
    names = NameMap()
    fname = os.path.join(DATA_DIR, 'preds.tsv')
    teams = {}
    with open(fname) as buff:
        for row in csv.DictReader(buff, delimiter="\t"):
            team_id = names.lookup(row["team_name"])
            win_pct = float(row["rd2_win"])
            region = row["team_region"]
            try:
                seed = int(row["team_seed"])
            except ValueError:
                seed = int(row["team_seed"][:2])
            if region not in teams:
                teams[region] = {}
            teams[region][seed] = {"team_id": team_id, "win_pct": win_pct}
    first_round_matchups = {}

    for region, region_data in teams.iteritems():
        for seed, seed_data in region_data.iteritems():
            opponent = teams[region][17 - seed]
            if opponent["team_id"] < seed_data["team_id"]:
                key = "2015_{:d}_{:d}".format(opponent["team_id"], seed_data["team_id"])
                first_round_matchups[key] = opponent["win_pct"]
            else:
                key = "2015_{:d}_{:d}".format(seed_data["team_id"], opponent["team_id"])
                first_round_matchups[key] = seed_data["win_pct"]

    team_lookup = id_to_name()
    data = []
    with open(os.path.join(DIR, 'out_data/predictions/2015.csv')) as buff:
        with open(os.path.join(DIR, 'out_data/predictions/2015_ensemble.csv'), 'w') as write_buff:
            write_buff.write("id,pred\n")
            reader = csv.DictReader(buff)
            for row in reader:
                # if "1246" in row["id"]:
                #     _, team_one, team_two = map(int, row["id"].split("_"))
                #     if team_one == 1246:
                #         write_buff.write("{:s},{:f}\n".format(row["id"], 0.9999))
                #     else:
                #         write_buff.write("{:s},{:f}\n".format(row["id"], 1 - 0.9999))
                if row["id"] in first_round_matchups:
                    _, team_one, team_two = map(int, row["id"].split("_"))
                    data.append([team_lookup[team_one], team_lookup[team_two], 100 * first_round_matchups[row["id"]],
                                 100 * float(row["pred"])])
                    write_buff.write("{:s},{:f}\n".format(row["id"], 0.5 * (float(row["pred"]) + first_round_matchups[row["id"]])))
                else:
                    write_buff.write("{:s},{:f}\n".format(row["id"], float(row["pred"])))



def main():
    data = DataHandler()
    data.build()


if __name__ == '__main__':
    human_readable()
