import heapq
import threading
from collections import defaultdict

import pandas as pd
from mpi4py import MPI


def data_process(data, n_threads, rank):
    threads_res = {'Followers': [], 'Friends': [], 'Mentions': defaultdict(int)}
    lock = threading.Lock()
    threads = []
    for i in range(n_threads):
        thread_data = data.iloc[data.shape[0] * i // n_threads:data.shape[0] * (i + 1) // n_threads]
        thread = threading.Thread(target=data_thread, args=(thread_data, threads_res, lock, rank, i))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    process_res = {'Followers': [], 'Friends': [], 'set_Followers': set(), 'set_Friends': set(),
                   'Mentions': threads_res['Mentions']}
    for task in ['Followers', 'Friends']:
        for elem in threads_res[f"{task}"]:
            if elem[1] not in process_res[f"set_{task}"]:
                process_res[f"set_{task}"].add(elem[1])
                process_res[f"{task}"].append(elem)
        process_res[f"{task}"].sort(key=lambda x: x[0], reverse=True)
        process_res[f"{task}"] = process_res[f"{task}"][:20]
    return process_res


def data_thread(data, threads_res, lock, n_process, n_thread):
    thread_res = {'Followers': [], 'Friends': [], 'Mentions': defaultdict(int), 'set_Followers': set(),
                  'set_Friends': set()}
    for index, row in data.iterrows():
        for task in ['Followers', 'Friends']:
            if row['Name'] not in thread_res[f'set_{task}']:
                if len(thread_res[f'{task}']) < 20:
                    heapq.heappush(thread_res[f'{task}'], (row[f'{task}'], row['Name']))
                    thread_res[f'set_{task}'].add(row['Name'])
                elif thread_res[f'{task}'][0][0] < row[f'{task}']:
                    deleted = heapq.heappop(thread_res[f'{task}'])
                    thread_res[f'set_{task}'].remove(deleted[1])
                    heapq.heappush(thread_res[f'{task}'], (row[f'{task}'], row['Name']))
                    thread_res[f'set_{task}'].add(row['Name'])
        if isinstance(row['UserMentionID'], str):
            for user in row['UserMentionID'].split(','):
                thread_res['Mentions'][user] += 1
    with lock:
        threads_res["Followers"].extend(thread_res["Followers"])
        threads_res["Friends"].extend(thread_res["Friends"])
        for user in thread_res["Mentions"].keys():
            threads_res['Mentions'][user] += thread_res['Mentions'][user]
        print(f"Finished process {n_process}, thread {n_thread}")


if __name__ == "__main__":
    data = pd.read_csv("FIFA.csv", encoding='cp1251', encoding_errors='ignore')
    n_threads = 4

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    data_to_process = data.iloc[data.shape[0] * rank // size:data.shape[0] * (rank + 1) // size]
    process_res = data_process(data_to_process, n_threads, rank)

    processes_res_raw = comm.gather(process_res, root=0)
    if rank == 0:
        processes_res = {'Followers': [elem for res in processes_res_raw for elem in res["Followers"]],
                         'Friends': [elem for res in processes_res_raw for elem in res["Friends"]],
                         'Mentions': defaultdict(int)}
        results = {'Followers': [], 'Friends': [], 'Mentions': [], 'set_Followers': set(), 'set_Friends': set()}
        for task in ['Followers', 'Friends']:
            for elem in processes_res[f"{task}"]:
                if elem[1] not in results[f"set_{task}"]:
                    results[f"set_{task}"].add(elem[1])
                    results[f"{task}"].append(elem)
            results[f"{task}"].sort(key=lambda x: x[0], reverse=True)
            results[f"{task}"] = results[f"{task}"][:20]
        for res in processes_res_raw:
            for key in res["Mentions"].keys():
                processes_res["Mentions"][key] += res["Mentions"][key]
        for key in processes_res["Mentions"].keys():
            if len(results["Mentions"]) < 20:
                heapq.heappush(results["Mentions"], (processes_res["Mentions"][key], key))
            elif results["Mentions"][0][0] < processes_res["Mentions"][key]:
                heapq.heappop(results["Mentions"])
                heapq.heappush(results["Mentions"], (processes_res["Mentions"][key], key))
        results["Mentions"].sort(key=lambda x: x[0], reverse=True)

        print('Followers')
        for elem in results["Followers"]:
            print(elem)

        print('Friends')
        for elem in results["Friends"]:
            print(elem)

        print('Mentions')
        for elem in results["Mentions"]:
            print(elem)