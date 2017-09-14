from __future__ import absolute_import

import sys
from tabulate import tabulate


def PrintProgress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s %s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()
    if count == total:
        sys.stdout.write("\n")


def PrintTable(data, headers):
    print tabulate(data, headers=headers, tablefmt='orgtbl')


def PrintModels(models):
    header_idx = {}
    combined_headers = []
    combined_data = []
    headers, datas = [], []
    for model in models:
        model_header, model_data = model.GetDebugTable()
        headers.append(model_header)
        datas.append(model_data)
        for header in model_header:
            if header_idx.get(header) is None:
                header_idx[header] = len(combined_headers)
                combined_headers.append(header)

    for index in range(len(datas)):
        data = datas[index]
        header = headers[index]
        current = ['N/A'] * len(combined_headers)
        for row_idx in range(len(header)):
            current_header = header[row_idx]
            current_data = data[row_idx]
            current[header_idx[current_header]] = current_data
        combined_data.append(current)

    PrintTable(combined_data, combined_headers)
