import sys
from os import path, makedirs
import json
from operator import itemgetter
from itertools import product

CONFIG_FILE = path.join(path.dirname(__file__), 'experiment_plan.json')
with open(CONFIG_FILE) as config_in:
    CONFIG = json.load(config_in)


def generate_configs():
    config_items = CONFIG.items()
    keys = map(itemgetter(0), config_items)
    values = map(itemgetter(1), config_items)
    configs = []
    for value_item in product(*values):
        configs.append({
            key: value
            for key, value in zip(keys, value_item)
        })
    return configs


def save_configs(in_configs, in_output_folder):
    if not path.exists(in_output_folder):
        makedirs(in_output_folder)
    for index, config in enumerate(in_configs):
        config_filename = path.join(in_output_folder, '{}.json'.format(index))
        with open(config_filename, 'w') as config_out:
            json.dump(config, config_out)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'Usage: {} <result_folder>'.format(path.basename(__file__))
        exit()
    save_configs(generate_configs(), sys.argv[1])
