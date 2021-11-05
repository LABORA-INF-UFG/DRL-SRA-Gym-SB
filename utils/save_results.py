import csv
import json
from json import JSONEncoder

import numpy


class SaveResults:

    def __init__(self):
        self.file_name = 'data/history.csv'
        self.row = []
        #clearing the file
        open(self.file_name, 'w').close()


    def append(self,nrow):
        self.row.append(nrow)

    def save(self):
        with open(self.file_name, 'a+') as file:
            writer = csv.writer(file)
            writer.writerow(self.row)
        self.row = []

    @staticmethod
    def save_history(data):
        with open('data/history.json', 'w') as outfile:
            json.dump(data, outfile, cls=NumpyArrayEncoder)

    @staticmethod
    def save_history_in_file(file, data):
        with open(file, 'w') as outfile:
            json.dump(data, outfile, cls=NumpyArrayEncoder)

    @staticmethod
    def load_history(file=''):
        fl = file if file != '' else 'data/history.json'
        with open(fl, "r") as read_file:
            data = json.load(read_file)
        return data

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)