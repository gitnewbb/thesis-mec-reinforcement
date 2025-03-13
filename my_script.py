from untitled import Qlearning, MakeZspace, MakeDspace, calZ as original_calZ, generate_H as original_generate_H
import numpy as np
import matplotlib.pyplot as plt
import collections
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from itertools import product
import tkinter as tk
from tkinter import filedialog
from sklearn.decomposition import PCA
from collections import deque
import os
import sys

def save_plot(fig, filename):
    """ 저장할 폴더 선택 창을 띄움 """
    SAVE_DIR = os.getcwd()
    save_path = os.path.join(SAVE_DIR, filename)
    fig.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")

# 수정된 generate_H 함수: 전체 총합은 유지, 각 UE에 작은 랜덤 변동 적용
def generate_H(n, c1=0, c2=0, small_variance=True):
    A = np.ones((n, 3))
    if not small_variance:
        A[:, 0] = np.random.randint(12, 16, size=n) + c1
        A[:, 1] = np.random.randint(2000, 2500, size=n) + c2
    else:
        base_file = 14 + c1
        total_file = base_file * n
        file_sizes = np.random.normal(base_file, 0.5, n)
        file_sizes = np.maximum(file_sizes, 0.1)
        file_sizes = file_sizes * (total_file / np.sum(file_sizes))
        
        base_comp = 2250 + c2
        total_comp = base_comp * n
        comp_amounts = np.random.normal(base_comp, 5, n)
        comp_amounts = np.maximum(comp_amounts, 0.1)
        comp_amounts = comp_amounts * (total_comp / np.sum(comp_amounts))
        
        A[:, 0] = file_sizes
        A[:, 1] = comp_amounts
    A[:, 2] = np.ones(n) * 3
    return A

# 기존 MakeDspace 함수
def MakeDspace(n, step):
    array = [0, 1]
    eta = np.array(list(product(array, repeat=n)))
    
    array2 = np.round(np.arange(0, 1+step, step), 4)
    theta = np.array(list(product(array2, repeat=n)))
    
    array3 = np.round(np.arange(0, 1+step, step), 4)
    beta = np.array(list(product(array3, repeat=n)))
    q = np.empty((0, 3, n))
    for i in range(eta.shape[0]):
        e = np.array(eta[i]).reshape(-1, n)
        t = np.unique(e * theta[np.sum(e*theta, axis=1) <= 1], axis=0)
        t = np.unique(e * t[np.sum(np.abs(e - np.ceil(t)), axis=1) == 0], axis=0)
        b = np.unique(e * beta[np.sum(e*beta, axis=1) <= 1], axis=0)
        b = np.unique(e * b[np.sum(np.abs(e - np.ceil(b)), axis=1) == 0], axis=0)
        if t.shape[0] * b.shape[0] != 0:
            tb = np.array(list(product(t, b)))
            p = np.zeros((t.shape[0]*b.shape[0], 3, n))
            p[:, 0, :] = e
            p[:, 1:3, :] = tb
        q = np.append(q, p, axis=0)
    return q

# 기존 MakeZspace 함수
def MakeZspace(step):
    E = np.array([0])
    comp = np.arange(0, 1+step, step)
    Z = np.array(list(product(E, comp)))
    return Z

# 수정된 calZ 함수: 계산된 Time를 np.sum()하여 스칼라 반환
def calZ(H, D, timestep):
    L = 200
    p = 1.2
    pw = 0.8
    g = 127 + 30 * np.log(L)
    W = 5 * 10**6
    noisedbm = -100
    noise = 10**(noisedbm/10) * 10**(-3)
    R = W * np.log2(1 + p*g/(noise**0.5))
    f_loc = 0.8 * 10**9
    f_mec = 6 * 10**9
    energy = 0
    Time = 0
    # 로컬 처리
    Time += (1 - D[0]) * H[:, 1] * 10**6 / f_loc
    energy += (1 - D[0]) * H[:, 1] * 10**6 * (f_loc**2) * (10**(-27))
    # MEC 전송
    Time += D[0] * H[:, 0] * 10**6 / (R * (1 - D[0] + D[1]))
    energy += D[0] * p * H[:, 0] * 10**6 / (R * (1 - D[0] + D[1]))
    # MEC 계산
    Time += D[0] * H[:, 1] * 10**6 / ((1 - D[0] + D[2]) * f_mec)
    energy += D[0] * pw * H[:, 1] * 10**6 / ((1 - D[0] + D[2]) * f_mec)
    z = np.zeros(2)
    z[0] = np.sum(energy)
    z[1] = 1 - np.sum(D[0] * D[2])
    # Time를 스칼라로 반환 (전체 합)
    Time = np.sum(Time)
    return z, Time

def createEpsilonGreedyPolicy(Q, epsilon, num_actions):
    def policyFunction(state):
        Action_probabilities = np.ones(num_actions, dtype=float) * epsilon / num_actions
        best_action = np.argmax(Q[state])
        Action_probabilities[best_action] += (1.0 - epsilon)
        return Action_probabilities
    return policyFunction

def Qlearning(Z, D, lr, dfactor, n, timestep, c1=0, c2=0, small_variance=True):
    """ Q-learning을 사용하여 상태-액션 Q 테이블을 학습하는 함수 (로스 폭발 방지) """
    
    Q = collections.defaultdict(lambda: np.zeros(D.shape[0]))  # Q 테이블 초기화
    epsilon = 0.01 + 0.1 / np.sqrt(n)  # UE 개수에 따라 탐색률 조정
    policy = createEpsilonGreedyPolicy(Q, epsilon, D.shape[0])  # Epsilon-Greedy Policy
    adaptive_lr = lr / np.sqrt(n)  # UE 개수 증가에 따라 학습률 감소

    for _ in range(2 * D.shape[0]):  # 학습 반복 횟수 증가
        for i, zt in enumerate(Z):  # 각 상태에서 시작하는 에피소드
            ti, totalT, nz = 0, 0, i
            z_l, q_l = [], []
            H = generate_H(n, c1, c2, small_variance=small_variance)  # 환경 변수 생성
            h = H * timestep / H[0][2]

            while ti < H[0][2]:  # 시간 한도 내에서 반복
                actionprob = policy(nz)
                avaD = (np.sum(D, axis=2)[:, 2] <= zt[1])  # 가능한 행동 필터링
                valid_actions = np.where(avaD)[0]

                if len(valid_actions) == 0:  # 가능한 행동이 없으면 종료
                    break

                action = np.random.choice(valid_actions, p=(actionprob[valid_actions] / np.sum(actionprob[valid_actions])))

                next_z, T = calZ(h, D[action], timestep)
                idx = np.where((next_z[1]-0.2 < Z[:, 1]) & (Z[:, 1] <= next_z[1]))[0]
                next_z_i = idx[0] if idx.size > 0 else len(Z) - 1

                # 보상 계산 및 Q 업데이트 (TD 오류 클리핑 적용)
                reward = -next_z[0]
                best_next_action = np.argmax(Q[next_z_i]) if np.any(Q[next_z_i]) else 0
                td_target = reward + dfactor * Q[next_z_i][best_next_action]
                td_delta = np.clip(td_target - Q[nz][action], -1, 1)  # TD 오류 크기 제한
                Q[nz][action] += adaptive_lr * td_delta  # 학습률 감소 적용

                # 경로 기록
                z_l.append(nz)
                q_l.append(action)
                totalT += T
                nz = next_z_i
                ti += timestep  # 시간 증가

            # 초과된 실행 시간 보상 패널티 (TD 오류 클리핑 적용)
            if totalT > H[0][2]:
                penalty = np.clip((totalT - H[0][2]) * adaptive_lr * 100, -1, 1)
                for z, a in zip(z_l, q_l):
                    Q[z][a] -= penalty

    return Q
def DQN(Z, D, lr, dfactor, n, timestep, c1=0, c2=0, use_pca=True):
    """
    Deep Q-Network (DQN) 학습 및 Q테이블 대체 모델 생성

    Args:
        Z (array): 상태 공간
        D (array): 행동 공간
        lr (float): 학습률
        dfactor (float): 할인율
        n (int): UE 수
        timestep (float): 시뮬레이션 시간 단위
        c1, c2 (float): 작업 부하 생성 파라미터
        use_pca (bool): 상태 공간 축소 여부

    Returns:
        dict: 학습된 DQN Q값 딕셔너리
    """

    # 📌 Q-learning을 사용하여 초기 Q-table 생성
    Q = Qlearning(Z, D, lr, dfactor, n, timestep, c1, c2)

    # 상태 공간 차원이 클 경우 PCA로 축소
    if use_pca and len(Z[0]) > 2:
        pca = PCA(n_components=2)
        Z_reduced = pca.fit_transform(Z)
    else:
        Z_reduced = Z

    state_size = Z_reduced.shape[1]
    action_size = D.shape[0]
    memory_size = 50000  # 경험 리플레이 크기 증가
    batch_size = 128
    epochs = 150  # 학습 횟수 증가
    learning_rate = lr / np.sqrt(n)  # UE 증가에 따른 학습률 조정

    # ✅ DQN 네트워크 구조 개선 (Batch Normalization 추가)
    model = Sequential([
        Dense(256, input_dim=state_size, activation='relu'),
        BatchNormalization(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dense(256, activation='relu'),
        Dense(action_size, activation='linear')
    ])
    model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))

    # 경험 리플레이 저장소
    memory = deque(maxlen=memory_size)

    # 📌 Q-table을 학습 데이터로 변환
    X_train = []
    y_train = []
    for state_idx in range(len(Z)):
        X_train.append(Z_reduced[state_idx])
        y_train.append(Q[state_idx])

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # ✅ Reward 정규화 적용 (Loss 폭발 방지)
    y_train = y_train / (1 + np.abs(y_train))

    print("Training DQN model...")
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)

    # Q값 예측 함수
    def predict_q(state_idx):
        state = Z_reduced[state_idx].reshape(1, -1)
        return model.predict(state, verbose=0)[0]

    if use_pca and len(Z[0]) > 2:
        return model, pca
    else:
        return model, None

# simulql_dqn_compare 함수: UE 수별 전략 비교
def simulql_dqn_compare(timestep, lr, step, dfactor, small_variance=True):
    n = 5  # 최대 UE 수
    Z = MakeZspace(step)
    
    Eresult_q = np.zeros(n)
    Tresult_q = np.zeros(n)
    Eresult_dqn = np.zeros(n)
    Tresult_dqn = np.zeros(n)
    Eresult_off = np.zeros(n)
    Tresult_off = np.zeros(n)
    Eresult_lo = np.zeros(n)
    Tresult_lo = np.zeros(n)
    
    c1 = 0
    c2 = 0
    
    for j in range(n):
        num_ues = j + 1
        print(f"Processing {num_ues} UEs...")
        D = MakeDspace(num_ues, step)
        print(f"Training Q-learning for {num_ues} UEs...")
        Q = Qlearning(Z, D, lr, dfactor, num_ues, timestep, c1, c2)
        print(f"Training DQN for {num_ues} UEs...")
        model,use_pca = DQN(Z, D, lr, dfactor, num_ues, timestep, c1, c2, small_variance)
        
        TL_q = 0; EL_q = 0
        TL_dqn = 0; EL_dqn = 0
        TL_off = 0; EL_off = 0
        TL_lo = 0; EL_lo = 0
        
        zi = 0
        zi_dqn = 0
        
        D_off = np.zeros((3, num_ues))
        D_off[0, :] = 1
        D_off[1, :] = 1/num_ues
        D_off[2, :] = 1/num_ues
        D_lo = np.zeros((3, num_ues))
        
        H = generate_H(num_ues, c1, c2, small_variance)
        for i in range(int(3/timestep)):
            h = H * timestep / H[0][2]
            
            avd_q = (np.sum(D, axis=2)[:, 2] <= Z[zi][1])
            action_q = np.argmax(Q[zi] - 100 * (1 - avd_q))
            next_z_q, T_q = calZ(h, D[action_q], timestep)
            TL_q += np.average(T_q)
            EL_q += next_z_q[0]
            idx = np.where((next_z_q[1]-0.2 < Z[:, 1]) & (Z[:, 1] <= next_z_q[1]))[0]
            zi = idx[0] if idx.size > 0 else len(Z)-1
            
            if use_pca and len(Z[0]) > 2:
                Z_reduced = pca.transform(Z)
            else:
                Z_reduced = Z

            state_dqn = Z_reduced[zi_dqn].reshape(1, -1)
            q_values = model.predict(state_dqn, verbose=0)[0]
            avd_dqn = (np.sum(D, axis=2)[:, 2] <= Z[zi_dqn][1])
            # 가능한 행동만 고려하여 Q값을 평가
            action_dqn = np.argmax(q_values - 100 * (1 - avd_dqn))
            next_z_dqn, T_dqn = calZ(h, D[action_dqn], timestep)
            TL_dqn += np.average(T_dqn)
            EL_dqn += next_z_dqn[0]
            idx = np.where((next_z_dqn[1]-0.2 < Z[:, 1]) & (Z[:, 1] <= next_z_dqn[1]))[0]
            zi_dqn = idx[0] if idx.size > 0 else len(Z)-1
            
            z_lo, T_lo = calZ(h, D_lo, timestep)
            TL_lo += np.average(T_lo)
            EL_lo += z_lo[0]
            
            z_off, T_off = calZ(h, D_off, timestep)
            TL_off += np.average(T_off)
            EL_off += z_off[0]
        
        Eresult_q[j] = EL_q
        Tresult_q[j] = TL_q
        Eresult_dqn[j] = EL_dqn
        Tresult_dqn[j] = TL_dqn
        Eresult_off[j] = EL_off
        Tresult_off[j] = TL_off
        Eresult_lo[j] = EL_lo
        Tresult_lo[j] = TL_lo
    
    # 에너지 소비량 그래프 저장
    fig1 = plt.figure(figsize=(10, 6))
    plt.plot(list(range(1, n+1)), Eresult_q, 'o-', label='Q-learning')
    plt.plot(list(range(1, n+1)), Eresult_dqn, 's-', label='DQN')
    plt.plot(list(range(1, n+1)), Eresult_lo, '^-', label='Local')
    plt.plot(list(range(1, n+1)), Eresult_off, 'x-', label='Offloading')
    plt.legend()
    plt.title(f'Energy Consumption (lr={lr}, dfactor={dfactor}, Small Variance={small_variance})')
    plt.xticks(list(range(1, n+1)))
    plt.xlabel('Number of UEs')
    plt.ylabel('Energy Consumption (J)')
    plt.grid(True, linestyle='--', alpha=0.7)
    save_plot(fig1, "energy_consumption_ues.png")
        
    # 연산 시간 그래프 저장
    fig2 = plt.figure(figsize=(10, 6))
    plt.plot(list(range(1, n+1)), Tresult_q, 'o-', label='Q-learning')
    plt.plot(list(range(1, n+1)), Tresult_dqn, 's-', label='DQN')
    plt.plot(list(range(1, n+1)), Tresult_lo, '^-', label='Local')
    plt.plot(list(range(1, n+1)), Tresult_off, 'x-', label='Offloading')
    plt.legend()
    plt.title(f'Computation Time (lr={lr}, dfactor={dfactor}, Small Variance={small_variance})')
    plt.xticks(list(range(1, n+1)))
    plt.xlabel('Number of UEs')
    plt.ylabel('Average Time (s)')
    plt.grid(True, linestyle='--', alpha=0.7)
    save_plot(fig2, "computation_time_ues.png")

        
    # 평균 에너지 및 연산 시간 비교 그래프 저장
    fig3 = plt.figure(figsize=(10, 8))

    # 에너지 비교 막대 그래프
    plt.subplot(2, 1, 1)
    bar_width = 0.35
    x = np.arange(n)
    plt.bar(x - bar_width/2, Eresult_q, bar_width, label='Q-learning')
    plt.bar(x + bar_width/2, Eresult_dqn, bar_width, label='DQN')
    plt.title('Average Energy Consumption')
    plt.xticks(x, [])
    plt.xlabel(' ')
    plt.ylabel('Energy (J)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')

    # 연산 시간 비교 막대 그래프
    plt.subplot(2, 1, 2)
    plt.bar(x - bar_width/2, Tresult_q, bar_width, label='Q-learning')
    plt.bar(x + bar_width/2, Tresult_dqn, bar_width, label='DQN')
    plt.title('Average Computation Time')
    plt.xticks(x, [])
    plt.xlabel(' ')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')

    plt.tight_layout()
    save_plot(fig3, "avg_energy_time_comparison_ues.png")
    
    return {
        'q_energy': Eresult_q,
        'dqn_energy': Eresult_dqn,
        'local_energy': Eresult_lo,
        'offload_energy': Eresult_off,
        'q_time': Tresult_q,
        'dqn_time': Tresult_dqn,
        'local_time': Tresult_lo,
        'offload_time': Tresult_off,
        'comp_labels': None
    }

# simulql_dqn_computation_compare 함수: 계산량 변화에 따른 비교
def simulql_dqn_computation_compare(timestep, lr, step, dfactor, small_variance=True):
    n = 3  # 고정 UE 수
    Z = MakeZspace(step)
    
    c1l = [-12, -8, -4, 0, 4]
    c2l = [-1500, -1000, -500, 0, 500]
    
    Eresult_q = np.zeros(len(c1l))
    Tresult_q = np.zeros(len(c1l))
    Eresult_dqn = np.zeros(len(c1l))
    Tresult_dqn = np.zeros(len(c1l))
    Eresult_off = np.zeros(len(c1l))
    Tresult_off = np.zeros(len(c1l))
    Eresult_lo = np.zeros(len(c1l))
    Tresult_lo = np.zeros(len(c1l))
    
    comp_labels = ['500~1000', '1000~1500', '1500~2000', '2000~2500', '2500~3000']
    
    for j in range(len(c1l)):
        c1 = c1l[j]
        c2 = c2l[j]
        D = MakeDspace(n, step)
        print(f"Training Q-learning for computation scenario {j+1}...")
        Q = Qlearning(Z, D, lr, dfactor, n, timestep, c1, c2)
        print(f"Training DQN for computation scenario {j+1}...")
        model,use_pca = DQN(Z, D, lr, dfactor, n, timestep, c1, c2, small_variance)
        
        TL_q = 0; EL_q = 0
        TL_dqn = 0; EL_dqn = 0
        TL_off = 0; EL_off = 0
        TL_lo = 0; EL_lo = 0
        
        zi = 0
        zi_dqn = 0
        
        D_off = np.zeros((3, n))
        D_off[0, :] = 1
        D_off[1, :] = 1/n
        D_off[2, :] = 1/n
        D_lo = np.zeros((3, n))
        
        H = generate_H(n, c1, c2, small_variance)
        for i in range(int(3/timestep)):
            h = H * timestep / H[0][2]
            
            avd_q = (np.sum(D, axis=2)[:, 2] <= Z[zi][1])
            action_q = np.argmax(Q[zi] - 100 * (1 - avd_q))
            next_z_q, T_q = calZ(h, D[action_q], timestep)
            TL_q += np.average(T_q)
            EL_q += next_z_q[0]
            idx = np.where((next_z_q[1]-0.2 < Z[:,1]) & (Z[:,1] <= next_z_q[1]))[0]
            zi = idx[0] if idx.size > 0 else len(Z)-1
            
            if use_pca and len(Z[0]) > 2:
                Z_reduced = pca.transform(Z)
            else:
                Z_reduced = Z

            state_dqn = Z_reduced[zi_dqn].reshape(1, -1)
            q_values = model.predict(state_dqn, verbose=0)[0]
            avd_dqn = (np.sum(D, axis=2)[:, 2] <= Z[zi_dqn][1])
            # 가능한 행동만 고려하여 Q값을 평가
            action_dqn = np.argmax(q_values - 100 * (1 - avd_dqn))
            next_z_dqn, T_dqn = calZ(h, D[action_dqn], timestep)
            TL_dqn += np.average(T_dqn)
            EL_dqn += next_z_dqn[0]
            idx = np.where((next_z_dqn[1]-0.2 < Z[:,1]) & (Z[:,1] <= next_z_dqn[1]))[0]
            zi_dqn = idx[0] if idx.size > 0 else len(Z)-1
            
            z_lo, T_lo = calZ(h, D_lo, timestep)
            TL_lo += np.average(T_lo)
            EL_lo += z_lo[0]
            
            z_off, T_off = calZ(h, D_off, timestep)
            TL_off += np.average(T_off)
            EL_off += z_off[0]
        
        Eresult_q[j] = EL_q
        Tresult_q[j] = TL_q
        Eresult_dqn[j] = EL_dqn
        Tresult_dqn[j] = TL_dqn
        Eresult_off[j] = EL_off
        Tresult_off[j] = TL_off
        Eresult_lo[j] = EL_lo
        Tresult_lo[j] = TL_lo
    
    # 에너지 소비량 그래프 저장
    fig1 = plt.figure(figsize=(10, 6))
    plt.plot(list(range(1, len(c1l)+1)), Eresult_q, 'o-', label='Q-learning')
    plt.plot(list(range(1, len(c1l)+1)), Eresult_dqn, 's-', label='DQN')
    plt.plot(list(range(1, len(c1l)+1)), Eresult_lo, '^-', label='Local')
    plt.plot(list(range(1, len(c1l)+1)), Eresult_off, 'x-', label='Offloading')
    plt.legend()
    plt.title(f'Energy Consumption by Computation Amount (lr={lr}, dfactor={dfactor}, UEs={n}, Small Variance={small_variance})')
    plt.xticks(list(range(1, len(c1l)+1)), labels=comp_labels)
    plt.xlabel('Computation Cycle of Data (Mbits)')
    plt.ylabel('Energy Consumption (J)')
    plt.grid(True, linestyle='--', alpha=0.7)
    save_plot(fig1, "energy_computation_amount.png")

    # 연산 시간 그래프 저장
    fig2 = plt.figure(figsize=(10, 6))
    plt.plot(list(range(1, len(c1l)+1)), Tresult_q, 'o-', label='Q-learning')
    plt.plot(list(range(1, len(c1l)+1)), Tresult_dqn, 's-', label='DQN')
    plt.plot(list(range(1, len(c1l)+1)), Tresult_lo, '^-', label='Local')
    plt.plot(list(range(1, len(c1l)+1)), Tresult_off, 'x-', label='Offloading')
    plt.legend()
    plt.title(f'Computation Time by Computation Amount (lr={lr}, dfactor={dfactor}, UEs={n}, Small Variance={small_variance})')
    plt.xticks(list(range(1, len(c1l)+1)), labels=comp_labels)
    plt.xlabel('Computation Cycle of Data (Mbits)')
    plt.ylabel('Average Time (s)')
    plt.grid(True, linestyle='--', alpha=0.7)
    save_plot(fig2, "computation_time_computation_amount.png")

    # 평균 에너지 및 연산 시간 비교 그래프 저장
    fig3 = plt.figure(figsize=(10, 8))

    # 에너지 비교 막대 그래프
    plt.subplot(2, 1, 1)
    bar_width = 0.35
    x = np.arange(len(c1l))
    plt.bar(x - bar_width/2, Eresult_q, bar_width, label='Q-learning')
    plt.bar(x + bar_width/2, Eresult_dqn, bar_width, label='DQN')
    plt.title('Average Energy Consumption')
    plt.xticks(x, comp_labels)
    plt.xlabel('Computation Cycle of Data (Mbits)')
    plt.ylabel('Energy (J)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')

    # 연산 시간 비교 막대 그래프
    plt.subplot(2, 1, 2)
    plt.bar(x - bar_width/2, Tresult_q, bar_width, label='Q-learning')
    plt.bar(x + bar_width/2, Tresult_dqn, bar_width, label='DQN')
    plt.title('Average Computation Time')
    plt.xticks(x, comp_labels)
    plt.xlabel('Computation Cycle of Data (Mbits)')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')

    plt.tight_layout()
    save_plot(fig3, "avg_energy_time_computation_amount.png")
    
    return {
        'q_energy': Eresult_q,
        'dqn_energy': Eresult_dqn,
        'local_energy': Eresult_lo,
        'offload_energy': Eresult_off,
        'q_time': Tresult_q,
        'dqn_time': Tresult_dqn,
        'local_time': Tresult_lo,
        'offload_time': Tresult_off,
        'comp_labels': comp_labels
    }

if __name__ == "__main__":
    set_step=0.2
    print("Running comparison with varying number of UEs and small variance in data...")
    results_ue = simulql_dqn_compare(timestep=0.1, lr=0.0001, step=set_step, dfactor=0.9, small_variance=True)
    
    print("\nRunning comparison with varying computation requirements and small variance in data...")
    results_comp = simulql_dqn_computation_compare(timestep=0.1, lr=0.0001, step=set_step, dfactor=0.9, small_variance=True)


    SAVE_DIR = os.path.dirname(os.path.abspath(__file__))  # 현재 실행된 파일의 폴더
    log_file_path = os.path.join(SAVE_DIR, "results_log.txt")
    with open(log_file_path, "w", encoding="utf-8") as log_file:
        sys.stdout = log_file  # print()를 파일에 저장하도록 변경
        
        # ✅ 기존 print() 출력 부분 (파일에 저장됨)
        print("\n--- Summary Statistics (Small Variance) ---")
        print("Average Energy Consumption:")
        print(f"  Q-learning: {np.mean(results_ue['q_energy']):.4f} J")
        print(f"  DQN: {np.mean(results_ue['dqn_energy']):.4f} J")
        print(f"  Local: {np.mean(results_ue['local_energy']):.4f} J")
        print(f"  Offloading: {np.mean(results_ue['offload_energy']):.4f} J")

        print("\nAverage Computation Time:")
        print(f"  Q-learning: {np.mean(results_ue['q_time']):.4f} s")
        print(f"  DQN: {np.mean(results_ue['dqn_time']):.4f} s")
        print(f"  Local: {np.mean(results_ue['local_time']):.4f} s")
        print(f"  Offloading: {np.mean(results_ue['offload_time']):.4f} s")

        print("\nRunning baseline comparison with original uniform distribution...")
        results_ue_baseline = simulql_dqn_compare(timestep=0.1, lr=0.0001, step=set_step, dfactor=0.9, small_variance=False)

        print("\n--- Comparison: Small Variance vs. Original ---")
        q_energy_improvement = (np.mean(results_ue_baseline['q_energy']) - np.mean(results_ue['q_energy'])) / np.mean(results_ue_baseline['q_energy']) * 100
        dqn_energy_improvement = (np.mean(results_ue_baseline['dqn_energy']) - np.mean(results_ue['dqn_energy'])) / np.mean(results_ue_baseline['dqn_energy']) * 100

        print(f"  Q-learning: {q_energy_improvement:.2f}% change with small variance")
        print(f"  DQN: {dqn_energy_improvement:.2f}% change with small variance")

    # 표준 출력을 원래대로 복원
    sys.stdout = sys.__stdout__

    print(f"✅ 로그 저장 완료: {log_file_path}")
