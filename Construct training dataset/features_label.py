# Preprocessing

import numpy as np
import pandas as pd

usage_cols = ['start time', 'end time', 'C', 'D', 'machine ID', 'mean CPU usage', 'memory usage', 'H', 'unmapped page cache',
              'total page cache', 'max memory', 'mean disk I/O', 'mean disk space', 'max CPU usage', 'max disk I/O', 'P', 'MAI', 'R', 'S', 'T']
max_CPU, mean_CPU, max_IO, mean_IO, mean_disk_space = [], [], [], [], []
max_memory, memory_usage, page_used, MAI = [], [], [], []
output_cols = ['start time', 'end time', 'machine ID', 'max CPU usage', 'mean CPU usage', 'max disk I/O time', 'mean disk I/O time',
               'max disk sapce usage', 'mean disk space usage', 'max memory usage', 'mean memory usage', 'max page cache', 'mean page cache', 'max MAI', 'mean MAI']

cur_x_index = 0
for part in range(155):
    x_df = pd.DataFrame(columns=['start time', 'end time', 'machine ID', 'max CPU usage', 'mean CPU usage', 'max disk I/O time', 'mean disk I/O time',
                                 'max disk sapce usage', 'mean disk space usage', 'max memory usage', 'mean memory usage', 'max page cache', 'mean page cache', 'max MAI', 'mean MAI'])
    input_file = '../task_usage_gzip/part-{:05d}-of-00500.csv.gz'.format(part)
    output_file = './processed_task_usage/part-{:05d}-of-00500.csv.gz'.format(
        part)
    print('Processing ' + input_file)

    usage_df = pd.read_csv(input_file, compression='gzip', names=usage_cols)
    usage_df = usage_df.drop(['C', 'D', 'H', 'P', 'R', 'S', 'T'], axis=1)
    x_df = usage_df.sort_values(by=['machine ID', 'start time'])
    x_df.to_csv(output_file, index=False, compression='gzip')

# Original To One Row

import pandas as pd
from collections import OrderedDict

for part in range(154, 155):
    input_file = './processed_task_usage/part-{:05d}-of-00500.csv.gz'.format(
        part)
    output_file = './tmp_results/part-{:05d}-of-00500.csv'.format(part)
    print('Processing ' + input_file)

    usage_df = pd.read_csv(input_file)
    length = int(len(usage_df) / 13)
    interval, cur_ID = 300000000, 5
    cur_start = int(usage_df['start time'][0] / interval)
    max_CPU, mean_CPU, max_IO, mean_IO, mean_disk_space = [], [], [], [], []
    max_memory, memory_usage, page_used, MAI = [], [], [], []
    x_dict = OrderedDict([('timestamp', []), ('machine ID', []), ('max CPU usage', []), ('mean CPU usage', []), ('max disk I/O', []), ('mean disk I/O', []), ('max disk space', []),
                          ('mean disk space', []), ('max memory usage', []), ('mean memory usage', []), ('max page cache', []), ('mean page cache', []), ('max MAI', []), ('mean MAI', [])])

    for i in range(length):
        if usage_df['start time'][i] >= (cur_start) * interval and usage_df['end time'][i] <= (cur_start + 1) * interval:
            if usage_df['max CPU usage'][i] > 0:
                max_CPU.append(usage_df['max CPU usage'][i])
            if usage_df['mean CPU usage'][i] > 0:
                mean_CPU.append(usage_df['mean CPU usage'][i])
            if usage_df['max disk I/O'][i] > 0:
                max_IO.append(usage_df['max disk I/O'][i])
            if usage_df['mean disk I/O'][i] > 0:
                mean_IO.append(usage_df['mean disk I/O'][i])
            if usage_df['mean disk space'][i] > 0:
                mean_disk_space.append(usage_df['mean disk space'][i])
            if usage_df['max memory'][i] > 0:
                max_memory.append(usage_df['max memory'][i])
            if usage_df['memory usage'][i] > 0:
                memory_usage.append(usage_df['memory usage'][i])
            if usage_df['total page cache'][i] > 0 and usage_df['unmapped page cache'][i] > 0:
                page_used.append(usage_df['total page cache'][
                                 i] - usage_df['unmapped page cache'][i])
            if usage_df['MAI'][i] > 0:
                MAI.append(usage_df['MAI'][i])
        elif usage_df['machine ID'][i] == cur_ID:
            #
            x_dict['timestamp'].append(cur_start)
            x_dict['machine ID'].append(usage_df['machine ID'][i - 1])
            if max_CPU:
                x_dict['max CPU usage'].append(sum(max_CPU))
            else:
                x_dict['max CPU usage'].append(0.0)
            if mean_CPU:
                x_dict['mean CPU usage'].append(sum(mean_CPU))
            else:
                x_dict['mean CPU usage'].append(0.0)
            if max_IO:
                x_dict['max disk I/O'].append(sum(max_IO))
            else:
                x_dict['max disk I/O'].append(0.0)
            if mean_IO:
                x_dict['mean disk I/O'].append(sum(mean_IO))
            else:
                x_dict['mean disk I/O'].append(0.0)
            if mean_disk_space:
                x_dict['max disk space'].append(max(mean_disk_space))
                x_dict['mean disk space'].append(sum(mean_disk_space))
            else:
                x_dict['max disk space'].append(0.0)
                x_dict['mean disk space'].append(0.0)
            if max_memory:
                x_dict['max memory usage'].append(sum(max_memory))
            else:
                x_dict['max memory usage'].append(0.0)
            if memory_usage:
                x_dict['mean memory usage'].append(
                    sum(memory_usage) / float(len(memory_usage)))
            else:
                x_dict['mean memory usage'].append(0.0)
            if page_used:
                x_dict['max page cache'].append(max(page_used))
                x_dict['mean page cache'].append(
                    sum(page_used) / float(len(page_used)))
            else:
                x_dict['max page cache'].append(0.0)
                x_dict['mean page cache'].append(0.0)
            if MAI:
                x_dict['max MAI'].append(max(MAI))
                x_dict['mean MAI'].append(sum(MAI) / float(len(MAI)))
            else:
                x_dict['max MAI'].append(0.0)
                x_dict['mean MAI'].append(0.0)

            max_CPU = [usage_df['max CPU usage'][i]] if usage_df[
                'max CPU usage'][i] > 0 else []
            mean_CPU = [usage_df['mean CPU usage'][i]] if usage_df[
                'mean CPU usage'][i] > 0 else []
            max_IO = [usage_df['max disk I/O'][i]
                      ] if usage_df['max disk I/O'][i] > 0 else []
            mean_IO = [usage_df['mean disk I/O'][i]
                       ] if usage_df['mean disk I/O'][i] > 0 else []
            mean_disk_space = [usage_df['mean disk space'][
                i]] if usage_df['mean disk space'][i] > 0 else []
            max_memory = [usage_df['max memory'][i]] if usage_df[
                'max memory'][i] > 0 else []
            memory_usage = [usage_df['memory usage'][i]] if usage_df[
                'memory usage'][i] > 0 else []
            page_used = [usage_df['total page cache'][i] - usage_df['unmapped page cache'][i]] if usage_df[
                'total page cache'][i] > 0 and usage_df['unmapped page cache'][i] > 0 else []
            MAI = [usage_df['MAI'][i]] if usage_df['MAI'][i] > 0 else []
            cur_start = int(usage_df['start time'][i] / interval)
        else:
            max_CPU = [usage_df['max CPU usage'][i]] if usage_df[
                'max CPU usage'][i] > 0 else []
            mean_CPU = [usage_df['mean CPU usage'][i]] if usage_df[
                'mean CPU usage'][i] > 0 else []
            max_IO = [usage_df['max disk I/O'][i]
                      ] if usage_df['max disk I/O'][i] > 0 else []
            mean_IO = [usage_df['mean disk I/O'][i]
                       ] if usage_df['mean disk I/O'][i] > 0 else []
            mean_disk_space = [usage_df['mean disk space'][
                i]] if usage_df['mean disk space'][i] > 0 else []
            max_memory = [usage_df['max memory'][i]] if usage_df[
                'max memory'][i] > 0 else []
            memory_usage = [usage_df['memory usage'][i]] if usage_df[
                'memory usage'][i] > 0 else []
            page_used = [usage_df['total page cache'][i] - usage_df['unmapped page cache'][i]] if usage_df[
                'total page cache'][i] > 0 and usage_df['unmapped page cache'][i] > 0 else []
            MAI = [usage_df['MAI'][i]] if usage_df['MAI'][i] > 0 else []

            cur_ID = usage_df['machine ID'][i]
            cur_start = int(usage_df['start time'][i] / interval)

    x_df = pd.DataFrame.from_dict(x_dict)
    x_df.to_csv(output_file, index=False)

from collections import OrderedDict

x_df = OrderedDict([('timestamp', []), ('machine ID', []), ('max CPU', [])])
x_df['timestamp'].append(1)
x_df['machine ID'].append(2)
x_df['max CPU'].append(3)
test = pd.DataFrame.from_dict(x_df)
test.to_csv('./test.csv', index=False)

import pandas as pd

for part in range(200, 236):
    usage_file = './processed_task_usage/part-{:05d}-of-00500.csv.gz'.format(
        part)
    x_file = './tmp_results/part-{:05d}-of-00500.csv'.format(part)
    out_file = './tmp_results/part-{:05d}-of-00500.csv'.format(part)
    print('Processing ' + usage_file)

    usage_df = pd.read_csv(usage_file)
    length = int(len(usage_df) / 13)
    x_df = pd.read_csv(x_file)

    mean_disk_space = []
    interval = 300000000
    cur_start, cur_ID, cur_index = int(
        usage_df['start time'][0] / interval), 5, 0
    for i in range(length):
        if usage_df['start time'][i] >= (cur_start) * interval and usage_df['end time'][i] <= (cur_start + 1) * interval:
            if usage_df['mean disk space'][i] > 0:
                mean_disk_space.append(usage_df['mean disk space'][i])
        elif usage_df['machine ID'][i] == cur_ID:
            if mean_disk_space:
                x_df.set_value(cur_index, 'max disk space',
                               max(mean_disk_space))
                x_df.set_value(cur_index, 'mean disk space',
                               sum(mean_disk_space))
            else:
                x_df.set_value(cur_index, 'max disk space', 0.0)
                x_df.set_value(cur_index, 'mean disk space', 0.0)
            mean_disk_space = [usage_df['mean disk space'][
                i]] if usage_df['mean disk space'][i] > 0 else []
            cur_start = int(usage_df['start time'][i] / interval)
            cur_index = cur_index + 1
        else:
            mean_disk_space = [usage_df['mean disk space'][
                i]] if usage_df['mean disk space'][i] > 0 else []
            cur_start = int(usage_df['start time'][i] / interval)
            cur_index = cur_index + 1

    x_df.to_csv(out_file, index=False)


# Get X Labels (Features)

import pandas as pd
IDs = pd.read_csv('./machine_id.csv')
machine_IDs = IDs['machine ID'].tolist()

output_file = './machine_label_X-500.csv'
start_pos = [0 for i in range(500)]

columns = ['timestamp', 'machine ID', 'max CPU usage', 'mean CPU usage', 'max disk I/O', 'mean disk I/O', 'max disk space',
           'mean disk space', 'max memory usage', 'mean memory usage', 'max page cache', 'mean page cache', 'max MAI', 'mean MAI']
out_df = pd.DataFrame(columns=columns)

for machine_index in range(500):
    print('Processing machine ID: ' + str(machine_IDs[machine_index]))
    for file_index in range(500):
        input_file = './tmp_results/part-{:05d}-of-00500.csv'.format(
            file_index)
        df = pd.read_csv(input_file)
        cur_start = start_pos[file_index]
        cur_end = cur_start
        while df['machine ID'][cur_end] == machine_IDs[machine_index]:
            cur_end = cur_end + 1
        #
        out_df = out_df.append(df[cur_start:cur_end])
        start_pos[file_index] = cur_end

out_df.to_csv(output_file, index=False)
