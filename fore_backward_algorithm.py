# 前向后向概率计算
import numpy as np


# 定义前向算法
def fore_algorithm(A, B, p_i, o, T, N):
    # 计算初值
    alpha = np.zeros((T, N))
    for i in range(N):
        h = o[0]
        alpha[0][i] = p_i[i] * B[i][h]
    # 递推
    for t in range(T - 1):
        h = o[t + 1]
        for i in range(N):
            a = 0
            for j in range(N):
                a += (alpha[t][j] * A[j][i])
            alpha[t + 1][i] = a * B[i][h]
    # 终止
    P = 0
    for i in range(N):
        P += alpha[T - 1][i]  # 状态概率
    return P, alpha


# 定义后向算法
def back_algorithm(A, B, p_i, o, T, N):
    # 设置初值，beta_t(i)=1
    beta = np.ones((T, N))
    # 递推
    for t in range(T - 1):
        t = T - t - 2
        h = o[t + 1]
        h = int(h)

        for i in range(N):
            beta[t][i] = 0
            for j in range(N):
                beta[t][i] += A[i][j] * B[j][h] * beta[t + 1][j]
    # 终止
    P = 0
    for i in range(N):
        h = o[0]
        h = int(h)
        P += p_i[i] * B[i][h] * beta[0][i]
    return P, beta


if __name__ == "__main__":
    T = 8
    N = 3
    A = [[0.5, 0.1, 0.4], [0.3, 0.5, 0.2], [0.2, 0.2, 0.6]]
    B = [[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]]
    pi = [0.2, 0.3, 0.5]
    O = ['红', '白', '红', '红', '白', '红', '白', '白']
    o = np.zeros(T, np.int)
    for i in range(T):
        if O[i] == '白':
            o[i] = 1
        else:
            o[i] = 0
    PF, alpha = fore_algorithm(A, B, pi, o, T, N)
    PB, beta = back_algorithm(A, B, pi, o, T, N)
    print("前向概率PF:", PF, "后向概率PB:", PB)
    # P = P(i_4=q_3|O,lambda)
    P = alpha[4 - 1][3 - 1] * beta[4 - 1][3 - 1]
    print("前向后向概率计算可得 P(i4=q3|O,lambda)=", P / PF)