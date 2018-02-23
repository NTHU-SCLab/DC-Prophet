import numpy as np
import pandas as pd
import math

MAX_TIME_INTERVAL = 8353

ylabel1 = 0
ylabel2 = 0
ylabel3 = 0

df = pd.read_csv(
    '/Users/chenhaoyun/Desktop/machine_events/part-00000-of-00001.csv', header=None)
df.columns = ['timestamp', 'machine ID', 'event type',
              'platform ID', 'capacity: CPU', 'capacity: memory']
df = df.drop(['platform ID'], 1)

# µs -> 5min (paper time interval)e
df['timestamp'] = df['timestamp'] // (300 * 1e6)

machine_series = df[df['timestamp'] == 0].drop(
    ['timestamp', 'event type', 'capacity: CPU', 'capacity: memory'], 1)

new_df = None
for index, row in machine_series.iterrows():
    if index % 500 == 0:
        new_df = pd.DataFrame(
            np.arange(0, MAX_TIME_INTERVAL + 1), columns=['timestamp'])
        new_df['machine ID'] = np.full(
            (MAX_TIME_INTERVAL + 1, 1), row['machine ID'])
        new_df['machine state'] = np.full((MAX_TIME_INTERVAL + 1, 1), 0)
        new_df['Y label'] = np.full((MAX_TIME_INTERVAL + 1, 1), 0)
        abnormal_state_df = df[
            (df['machine ID'] == row['machine ID']) & (df['event type'] <= 1)]

        now_list = []
        flag1 = False
        for index_ab, row_ab in abnormal_state_df.iterrows():
            if row_ab['event type'] == 1:
                iloc_record = row_ab['timestamp'].astype(int)
                flag1 = True
                new_df['machine state'].iloc[iloc_record] = 1
                now_list.append(iloc_record)
            elif flag1 == True:
                # REMOVE and ADD occur in one interval
                if ((row_ab['timestamp'].astype(int)) - iloc_record) <= 1:
                    flag1 = False
                else:
                    for i in range(1, (row_ab['timestamp'].astype(int)) - iloc_record):
                        new_df['machine state'].iloc[iloc_record + i] = 1
                        now_list.append(iloc_record + i)
                    flag1 = False

        now_list_counter = 0
        for i in range(len(now_list)):
            if i == (len(now_list) - 1):
                # FD Fail
                if (flag1 == True) and ((MAX_TIME_INTERVAL - now_list[i]) > 6):
                    new_df['Y label'].iloc[
                        now_list[i] - (now_list_counter)] = 3
                    for j in range(1, MAX_TIME_INTERVAL - now_list[i] + 1):
                        new_df['Y label'].iloc[now_list[i] + j] = -1
                    now_list_counter = 0
                elif now_list_counter <= 6:  # IR Fail
                    new_df['Y label'].iloc[
                        now_list[i] - (now_list_counter)] = 1
                    for j in range(1, now_list_counter + 1):
                        new_df['Y label'].iloc[
                            now_list[i] - (now_list_counter) + j] = -1
                    now_list_counter = 0
                else:  # SR Fail
                    new_df['Y label'].iloc[
                        now_list[i] - (now_list_counter)] = 2
                    for j in range(1, now_list_counter + 1):
                        new_df['Y label'].iloc[
                            now_list[i] - (now_list_counter) + j] = -1
                    now_list_counter = 0
            elif (now_list[i + 1] - now_list[i]) == 1:
                now_list_counter += 1
            else:
                if now_list_counter <= 6:  # IR Fail
                    new_df['Y label'].iloc[
                        now_list[i] - (now_list_counter)] = 1
                    for j in range(1, now_list_counter + 1):
                        new_df['Y label'].iloc[
                            now_list[i] - (now_list_counter) + j] = -1
                    now_list_counter = 0
                else:  # SR Fail
                    new_df['Y label'].iloc[
                        now_list[i] - (now_list_counter)] = 2
                    for j in range(1, now_list_counter + 1):
                        new_df['Y label'].iloc[
                            now_list[i] - (now_list_counter) + j] = -1
                    now_list_counter = 0

    else:
        tmp_df = pd.DataFrame(
            np.arange(0, MAX_TIME_INTERVAL + 1), columns=['timestamp'])
        tmp_df['machine ID'] = np.full(
            (MAX_TIME_INTERVAL + 1, 1), row['machine ID'])
        tmp_df['machine state'] = np.full((MAX_TIME_INTERVAL + 1, 1), 0)
        tmp_df['Y label'] = np.full((MAX_TIME_INTERVAL + 1, 1), 0)
        abnormal_state_df = df[
            (df['machine ID'] == row['machine ID']) & (df['event type'] <= 1)]

        now_list = []
        flag1 = False
        for index_ab, row_ab in abnormal_state_df.iterrows():
            if row_ab['event type'] == 1:
                iloc_record = row_ab['timestamp'].astype(int)
                flag1 = True
                tmp_df['machine state'].iloc[iloc_record] = 1
                now_list.append(iloc_record)
            elif flag1 == True:
                # REMOVE and ADD occur in one interval
                if ((row_ab['timestamp'].astype(int)) - iloc_record) <= 1:
                    flag1 = False
                else:
                    for i in range(1, (row_ab['timestamp'].astype(int)) - iloc_record):
                        tmp_df['machine state'].iloc[iloc_record + i] = 1
                        now_list.append(iloc_record + i)
                    flag1 = False

        now_list_counter = 0
        for i in range(len(now_list)):
            if i == (len(now_list) - 1):
                # FD Fail
                if (flag1 == True) and ((MAX_TIME_INTERVAL - now_list[i]) > 6):
                    tmp_df['Y label'].iloc[
                        now_list[i] - (now_list_counter)] = 3
                    for j in range(1, MAX_TIME_INTERVAL - now_list[i] + 1):
                        new_df['Y label'].iloc[now_list[i] + j] = -1
                    now_list_counter = 0
                elif now_list_counter <= 6:  # IR Fail
                    tmp_df['Y label'].iloc[
                        now_list[i] - (now_list_counter)] = 1
                    for j in range(1, now_list_counter + 1):
                        tmp_df['Y label'].iloc[
                            now_list[i] - (now_list_counter) + j] = -1
                    now_list_counter = 0
                else:  # SR Fail
                    tmp_df['Y label'].iloc[
                        now_list[i] - (now_list_counter)] = 2
                    for j in range(1, now_list_counter + 1):
                        tmp_df['Y label'].iloc[
                            now_list[i] - (now_list_counter) + j] = -1
                    now_list_counter = 0
            elif (now_list[i + 1] - now_list[i]) == 1:
                now_list_counter += 1
            else:
                if now_list_counter <= 6:  # IR Fail
                    tmp_df['Y label'].iloc[
                        now_list[i] - (now_list_counter)] = 1
                    for j in range(1, now_list_counter + 1):
                        tmp_df['Y label'].iloc[
                            now_list[i] - (now_list_counter) + j] = -1
                    now_list_counter = 0
                else:  # SR Fail
                    tmp_df['Y label'].iloc[
                        now_list[i] - (now_list_counter)] = 2
                    for j in range(1, now_list_counter + 1):
                        tmp_df['Y label'].iloc[
                            now_list[i] - (now_list_counter) + j] = -1
                    now_list_counter = 0

        frames = [new_df, tmp_df]
        new_df = pd.concat(frames, ignore_index=True)

        if index % 500 == 499:
            csv_file = 'machine_label_Y-{}.csv'.format(index + 1)
            new_df.to_csv(csv_file, encoding='utf-8', index=False)

            ylabel1 += len(new_df[new_df['Y label'] == 1])
            ylabel2 += len(new_df[new_df['Y label'] == 2])
            ylabel3 += len(new_df[new_df['Y label'] == 3])
            print(ylabel1)
            print(ylabel2)
            print(ylabel3)
            print('------------------------')

            new_df = None

        if index == 12476:
            csv_file = 'machine_label_Y-{}.csv'.format(index + 1)
            new_df.to_csv(csv_file, encoding='utf-8', index=False)

            ylabel1 += len(new_df[new_df['Y label'] == 1])
            ylabel2 += len(new_df[new_df['Y label'] == 2])
            ylabel3 += len(new_df[new_df['Y label'] == 3])
            print(ylabel1)
            print(ylabel2)
            print(ylabel3)
            print('END')

'''
246
120
7
------------------------
468
168
8
------------------------
838
251
9
------------------------
1085
318
17
------------------------
1285
416
19
------------------------
1506
505
21
------------------------
1755
642
22
------------------------
1994
707
24
------------------------
2228
854
28
------------------------
2437
921
28
------------------------
2596
1005
32
------------------------
2787
1105
34
------------------------
2977
1178
38
------------------------
3173
1289
40
------------------------
3416
1391
43
------------------------
3643
1487
48
------------------------
3903
1605
50
------------------------
4183
1716
59
------------------------
4449
1831
62
------------------------
4686
1959
67
------------------------
4924
2095
74
------------------------
5159
2222
80
------------------------
5422
2382
82
------------------------
5704
2448
85
------------------------
5924
2583
88
END
'''
