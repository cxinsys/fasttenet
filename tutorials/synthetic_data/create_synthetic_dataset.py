import sys
import os
import os.path as osp
import random
import pickle
import argparse
import string
import json

from tqdm import tqdm
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# sys.setrecursionlimit(5000)

def make_unidirectional_powerlaw_graph(n, m):
    G = nx.barabasi_albert_graph(n, m)
    
    # DG = nx.DiGraph()

    # for u, v in G.edges():
    #     if random.choice([True, False]):
    #         DG.add_edge(u, v)
    #     else:
    #         DG.add_edge(u, v)

    return G

def alphabetize_numbers(arr):
    def to_alphabetic(num):
        result = ''
        while num >= 0:
            num, remainder = divmod(num, 26)
            result = chr(65 + remainder) + result
            if num == 0:
                break
        return result

    return [to_alphabetic(int(item) - 1) for item in arr]

def generate_alphabetical_dict(N):
    alphabet = string.ascii_uppercase
    name_dict = {}
    
    def num_to_name(num):
        name = ""
        while num > 0:
            num -= 1
            name = alphabet[num % 26] + name
            num //= 26
        return name
    
    for i in range(1, N + 1):
        name_dict[i - 1] = num_to_name(i)
    
    return name_dict

def normalize(A, norm_in=True, norm_out=True):
    # Check whether A is a square matrix
    if A.shape[0] != A.shape[1]:
        raise ValueError(
            "The A (adjacency matrix) should be square matrix.")

    # Build propagation matrix (aka. transition matrix) _W from A
    W = A.copy()

    # Norm. out-degree
    if norm_out == True:
        sum_col_A = np.abs(A).sum(axis=0)
        sum_col_A[sum_col_A == 0] = 1
        if norm_in == False:
            Dc = 1 / sum_col_A
        else:
            Dc = 1 / np.sqrt(sum_col_A)
        # end of else
        W = Dc * W  # This is not matrix multiplication

    # Norm. in-degree
    if norm_in == True:
        sum_row_A = np.abs(A).sum(axis=1)
        sum_row_A[sum_row_A == 0] = 1
        if norm_out == False:
            Dr = 1 / sum_row_A
        else:
            Dr = 1 / np.sqrt(sum_row_A)
        # end of row
        W = np.multiply(W, np.mat(Dr).T)
        # Converting np.mat to ndarray
        # does not cost a lot.
        W = W.A
    # end of if
    """
    The normalization above is the same as the follows:
    >>> np.diag(Dr).dot(A.dot(np.diag(Dc)))
    """
    return W

def make_unique_pattern(min, max, time_points, n_values, num_segment):
    segment_size = time_points // (n_values * num_segment)
    
    # Selecting n values evenly spaced between 0 and 100
    up_down_values = []
    
    for i in range(num_segment):
        if i % 2 == 0: # even
            evenly_spaced_values = np.linspace(min, max, n_values)
        else:
            evenly_spaced_values = np.linspace(max, min, n_values)
        
        up_down_values.append(evenly_spaced_values)
        
    up_down_values = np.concatenate(up_down_values)
    
    # Creating an array to hold the time series inference
    time_series_data_even = np.zeros(time_points)
    
    # Splitting the 1000 time points into n equal segments and filling them with the selected values
    for i in range(n_values * num_segment):
        start_idx = i * segment_size
        end_idx = start_idx + segment_size if i != n_values-1 else time_points
        time_series_data_even[start_idx:end_idx] = up_down_values[i]
    
    return f"descrete-{n_values}-{num_segment}", time_series_data_even

def generate_lognormal_data(mean, sigma, size):
    data = np.random.lognormal(mean, sigma, size)
    
    return f"lognormal-{mean}-{sigma}", data

def make_connective_dg(G, source, top, DG, draws=[]):
    '''
    1. top 노드를 선택
    2. 이웃 노드 중 하나로 이동
    3. 부모와 자식을 잇는다.
    4. 이웃 노드를 구한 후 top 노드와의 최단경로가 자신 이상인 모든 이웃노드에 대해 2.부터 반복한다.
    5. 모든 적합한 이웃이 없는 노드(leaf)까지 순회한 후 종료.
    '''
    
    # print("source: ", source)
    
    e1s = list(G.neighbors(source)) # top의 이웃노드
    
    # print("e1s: ", e1s)
    
    e2s = []
    
    ss = nx.shortest_path_length(G, source=top, target=source)
    
    # print("ss: ", ss)
    
    for e1 in e1s:
        ts = nx.shortest_path_length(G, source=top, target=e1)
        
        # print("e1: ", e1)
        # print("ts: ", ts)
        
        if ss < ts:
            e2s.append(e1)
        elif ss == ts:
            draws.append((source, e1))
    
    # print("e2s: ", e2s)
    # print("draws: ", draws)
    
    if len(e2s)<=0:
        return DG, draws

    
    for e2 in e2s:
        DG.add_edge(source, e2)
        DG, draws = make_connective_dg(G, e2, top, DG, draws=draws)
        
    return DG, draws

def check_strong_connectivity(graph, start_node):
    # 모든 노드가 start_node로부터 접근 가능한지 확인
    reachable_nodes = nx.descendants(graph, start_node)
    
    # 자신도 포함해서 모든 노드에 도달 가능한지 확인
    reachable_nodes.add(start_node)
    
    result = len(reachable_nodes) == len(graph.nodes)

    print(f"Node {start_node} can reach all other nodes: {result}")
    
n = 20

es = [1]
ts = [512]

ds = [[0, int(n*0.02)], [int(n*0.02), int(n*0.15)], [int(n*0.15), n+1]]

droot = './datasets'

if __name__ == "__main__":
    
    n = 20 # 128, 256, 512, 1024, 2048
    e = 2
    t = 512 # 512, 1024, 2048, 4096, 8192
    dr = 1 # int(0.0004 * n)
    n_values = 16
    num_segment = 1

    G = nx.barabasi_albert_graph(n, e)
    DG2 = nx.DiGraph()
    
    # 노드 이름 알파벳 매핑
    mapping = generate_alphabetical_dict(n)
    # G = nx.relabel_nodes(G, mapping)
    
    n_degrees = sorted(G.degree, key=lambda x: x[1], reverse=True)
    
    DG2 = nx.DiGraph()
    # print(len(DG2.edges()))
    
    # 메인 노드 결정
    main_node_inf = random.choice(n_degrees[:dr])
    main_node = main_node_inf[0]
    
    DG2, draw_degrees = make_connective_dg(G, main_node, main_node, DG2, draws=[])
    draw_degrees = list(set(tuple(sorted(item)) for item in draw_degrees))
    
    for draw in draw_degrees:
        DG2.add_edge(draw[0], draw[1])
        
    check_strong_connectivity(DG2, main_node)
        
    W = nx.to_numpy_array(DG2, nodelist=list(sorted(DG2.nodes()))).T
    # W = normalize(W).T
    
    DG2 = nx.relabel_nodes(DG2, mapping)
    
    main_node = mapping[main_node]
    
    
    fpath = osp.join(droot, f'./synthetic_nodes-{n}_edges-{len(DG2.edges())}_top_degree-{dr}_time-{t}_values-{n_values}_segment-{num_segment}')
    os.makedirs(fpath, exist_ok=True)
        
    # 가장 긴 경로 찾기
    longest_path_length = nx.dag_longest_path_length(DG2)
    # longest_path_length = 20
    # 엣지가 많아지면 순환하는 경로가 생겨 longest path 함수에 오류가 발생하므로 임의로 딜레이를 주기로 함
    
        
    # 모든 노드에 최소 한 번 영향을 끼치도록 시간 추가
    ob_time = t + longest_path_length # 엣지가 많아지면 순환하는 경로가 생겨 longest path 함수에 오류가 발생하므로 임의로 딜레이를 주기로 함
    
    # 노드 초깃값 설정
    x0 = np.random.randint(-100, high=100, size=n)
    b = np.random.randint(0, high=10, size=n)

    a = np.random.uniform(0.7, 0.9, size=1)
    
    # 시간 + 노드 개수만큼 배열 생성
    trj = np.zeros((ob_time, n), dtype=np.float32)
        
    # 메인 노드 결정
    main_node_num = [k for k, v in mapping.items() if v == main_node][0]
    
    # 메인 시그널 결정
    name, main_signal = make_unique_pattern(min=0, max=1000, time_points=ob_time, n_values=n_values, num_segment=num_segment)
    
    cell_name = []

    for j in tqdm(range(ob_time)):
        # 메인 노드가 1개인 경우임.
        if j==0:
            trj[j, :] = x0
            trj[j, main_node_num] = main_signal[j]
            # trj[j, :] = a*np.dot(W, trj[j, :]) + (1-a)*b
            trj[j, :] = a*np.dot(W, trj[j, :]) + (1-a)*b + np.random.normal(0, 1, size=x0.shape)
            trj[j, main_node_num] = main_signal[j] + np.random.normal(0, 1, size=1)
            # trj[j, main_node] = main_signal[j]
        
        else:            # trj[j, :] = a*np.dot(W, trj[j-1, :]) + (1-a)*b
            trj[j, :] = a*np.dot(W, trj[j-1, :]) + (1-a)*b + np.random.normal(0, 1, size=x0.shape)
            trj[j, main_node_num] = main_signal[j] + np.random.normal(0, 1, size=1)
            # trj[j, main_node] = main_signal[j]
        

    # 모든 노드에 신호가 돈 후 시간만 가져오기
    trj = trj[longest_path_length:, :]
    
    # csv 형식을 위한 cellname
    for j in range(t):
        cell_name.append(f'cell_{j}')
        
    # 노드 이름 생성
    node_name = np.array(list(G.nodes()), dtype=str)

    # print(f'Node names: {node_name}')
    
    out_degrees = sorted(DG2.out_degree, key=lambda x: x[1], reverse=True)
    
    data = np.vectorize('{:.{}f}'.format)(trj, 4).T
    
    name_data = np.concatenate((node_name.reshape(-1, 1), data), axis=1)        
        
    cell_name = np.array(cell_name)
    blank = np.array([''])

    cell_name = np.concatenate((blank, cell_name)).reshape(1, -1)

    csv_data = np.concatenate((cell_name, name_data), axis=0)
    
    with open(osp.join(fpath, './synex_graph.json'), "w") as f:
        json.dump(nx.node_link_data(DG2), f)        

    np.savetxt(osp.join(fpath, f'./synex_main.csv'), csv_data.T, fmt='%s',
                delimiter=',')
    np.save(osp.join(fpath, f'./synex_main.npy'), trj.T)
    np.save(osp.join(fpath, f'./synex_main' + '_node_name.npy'), node_name)
    np.savetxt(osp.join(fpath, f'./synex_main' + ".outdegrees.txt"), out_degrees,
                fmt="%s")

    cell_sellect = np.ones(t, dtype=np.int32)
    np.savetxt(osp.join(fpath, './cell_select.txt'), cell_sellect, fmt='%d', delimiter='\t',
                encoding='utf-8')

    pseudo_time = np.arange(t, dtype=np.int32)
    np.savetxt(osp.join(fpath, './pseudo_time.txt'), pseudo_time, fmt='%d', delimiter='\t',
                encoding='utf-8')
    
    axis_font = {'fontname': 'Arial', 'size': '26'}
    title_font = {'fontname': 'Arial', 'size': '30'}
    
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    node_colors = ['white' if node != main_node else 'red' for node in DG2.nodes]
    # font_colors = ['black' if node != main_node else 'lightcolr' for node in DG2.nodes]
    pos=nx.spring_layout(DG2, seed=39,  k=1.2)
    
    nx.draw_networkx_nodes(DG2, pos, ax=ax2, 
                       node_color=node_colors,  # 노드 색상
                       edgecolors='black',  # 노드 엣지 색
                       node_size=2500,
                       linewidths=5, 
                       )  # 노드 크기

    nx.draw_networkx_edges(DG2, pos, ax=ax2,
                           width=2.5,  # 엣지 두께
                           arrowstyle='-|>',  # 화살표 스타일
                           arrowsize=40,  # 화살표 크기
                           node_size=2500,  # 노드 크기에 맞춰 화살표 간격 조정
                           min_source_margin=15,  # 소스 노드로부터 화살표 위치 조정
                           min_target_margin=15)  # 타겟 노드로부터 화살표 위치 조정)

    nx.draw_networkx_labels(DG2, pos, 
                        labels={node: node for node in DG2.nodes},
                        font_color='black',
                        font_size=35, )

    # 메인 노드에 대한 텍스트 색을 따로 지정
    nx.draw_networkx_labels(DG2, pos, 
                            labels={main_node: main_node},  # 메인 노드만 라벨링
                            font_color='white',
                            font_size=35, )  # 메인 노드 텍스트 색상
    
    import matplotlib.patches as mpatches
    main_node_patch = mpatches.Patch(color='red', label='main node')
    plt.legend(handles=[main_node_patch], loc='lower right', prop={'family': 'Arial', 'size': 17})
    # plt.show()
        
    fig2.savefig(osp.join(fpath, f'synex_graph.png'), dpi=300)
    
    plt.cla()   # clear the current axes
    plt.clf()   # clear the current figure
    plt.close() # closes the current figure
    
    
    plt.figure(figsize=(12, 6))
    plt.plot(trj, lw=2)
    plt.plot(trj[:, main_node_num], lw=2, color='red', label='main node')
    # plt.xlim(-0.5, 8+0.05)
    plt.yticks(font='Arial', fontsize=26)
    plt.tick_params(pad=10)
    plt.xticks(font='Arial', fontsize=26)
    plt.xlabel('Time', fontdict=axis_font, labelpad=10)
    plt.ylabel('Expression value', fontdict=axis_font, labelpad=13)
    plt.tight_layout()
    plt.legend(loc='lower right', prop={'family': 'Arial', 'size': 24})
    # plt.show()
    plt.savefig(osp.join(fpath, f'synex_expression.png'), dpi=300)
    
    pickle.dump(DG2, open(osp.join(fpath, f'synex_main_graph.pkl'), 'wb'))
    
    
    plt.cla()   # clear the current axes
    plt.clf()   # clear the current figure
    plt.close() # closes the current figure
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(trj[:, main_node_num], lw=2, color='red')
    plt.yticks(font='Arial', fontsize=26)
    plt.tick_params(pad=10)
    plt.xticks(font='Arial', fontsize=26)
    plt.xlabel('Time', fontdict=axis_font, labelpad=10)
    plt.ylabel('Expression value', fontdict=axis_font, labelpad=13)
    plt.tight_layout()
    # plt.show()
    plt.savefig(osp.join(fpath, f'synex_main_expression.png'), dpi=300)
    
    plt.cla()   # clear the current axes
    plt.clf()   # clear the current figure
    plt.close() # closes the current figure
    
    # # Creating a DataFrame to display the time series inference
    # time_series_df_even = pd.DataFrame({'Time': np.arange(1, ob_time+1), 'Value': main_signal})

    # # Plotting the new time series inference with evenly spaced values
    # plt.figure(figsize=(10, 6))
    # plt.plot(time_series_df_even['Time'], time_series_df_even['Value'], marker='o', linestyle='-', color='g')
    # plt.title('Time Series Data (Evenly Spaced Values)')
    # plt.xlabel('Time')
    # plt.ylabel('Value')
    # plt.grid(True)
    # plt.show()

    
    n_bins = np.ceil((main_signal.max() - main_signal.min()) / (0.5 * main_signal.std())).astype(np.int32)
    
    print("Number of Bins after binning for main signal: ", n_bins)
    
    with open(osp.join(fpath, 'synex_info.txt'), 'w', encoding='utf-8') as f:
        f.write(f'[Nodes] {n} \n')
        f.write(f'[Edges] {e} \n')
        f.write(f'[Top Degree Rank Range] {dr}\n')
        f.write(f'[Main Node] {main_node}, [Outdegrees] {main_node_inf[1]} \n')
        f.write(f'[Times] {t} \n')
        f.write(f'[Num. values]  {n_values}\n')
        f.write(f'[Num. segment] {num_segment}\n')
        f.write(f'[N_bins of main signal]  {n_bins}\n')
    
