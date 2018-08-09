import numpy as np
import pandas as pd
from _2048_ import game,init

MAX_EPISODES= 100
ACTIONS = [0,1,2,3]
GAMMA = 0.9
LR = 0.1
EPSILON = 0.9

def build_q_table():
    table = pd.DataFrame(
    columns = ACTIONS,
    )
    return table

def check_state_exist(table,state):
    if state not in table.index:
        table = table.append(
        pd.Series(
        [0]*len(ACTIONS),
        index = table.columns,
        name = state,
        )
        )
    return table

def choose_action(state, q_table):
    state_actions = q_table.loc[state, :]  
    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.argmax()
    return action_name

def learning(res,S,A,R,table):
    q_predict = table.loc[str(S),A]
    if res[0]:
        q_target = R + GAMMA*table.loc[str(res[1]),:].max()
    else:
        q_target = R
    table.loc[str(S),A] += LR*(q_target - q_predict)
    return table

def rl():
    q_table = build_q_table()
    for episode in range(MAX_EPISODES):
        res,R = init()
        S = res[1]
        q_table = check_state_exist(q_table,str(S))
        while res[0]:
            A = choose_action(str(S),q_table)
            res,R = game(A,S,R)
            S_ = res[1]
            q_table = check_state_exist(q_table,str(S_))
            q_table = learning(res,S,A,R,q_table)
            S = res[1]
    print('game over')

if __name__ == '__main__':
    rl()




