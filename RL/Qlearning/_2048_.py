#!/user/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'LTstrange'

import random
import copy
import time


def add_number():#随机选择添加数字(finished
    number_type = random.randint(0,10)
    if number_type == 0:
        add_num = 4
    else:
        add_num = 2
    return add_num

def add_num(field):
    l = []
    for ind_y,y in enumerate(field):
        for ind_x,x in enumerate(y):
            if x == 0:
                l.append([ind_y,ind_x])
    position = l[random.randint(0,len(l)-1)]
    field[position[0]][position[1]] = add_number()
    return field

def print_screen(field,score):#打印屏幕(finished
    print('--------------------')
    print('your score is:',score)
    print('\t%d\t%d\t%d\t%d\n'%(field[0][0],field[0][1],field[0][2],field[0][3]))
    print('\t%d\t%d\t%d\t%d\n'%(field[1][0],field[1][1],field[1][2],field[1][3]))
    print('\t%d\t%d\t%d\t%d\n'%(field[2][0],field[2][1],field[2][2],field[2][3]))
    print('\t%d\t%d\t%d\t%d\n'%(field[3][0],field[3][1],field[3][2],field[3][3]))

def init():
    field = [[0 for i in range(4)] for x in range(4)]
    a,b = [random.randint(0,3) , random.randint(0,3)],[random.randint(0,3) , random.randint(0,3)]
    field[a[0]][a[1]] = add_number()
    field[b[0]][b[1]] = add_number()
    return [True,field],0


def in_ope(field):
    while True:
        ope = input("你的操作是：\n（wasd）\n使用“exit()”来退出\n")
        if ope == 'w':
            return 0
        if ope == 'd':
            return 1
        if ope == 's':
            return 2
        if ope == 'a':
            return 3
        if ope == 'exit()':
            exit()
        else:
            print('please input w,s,a,d\nmeans up,down,left,right')
            continue

def Fl2r(obj):#转换field方向（左到右）(finished
    for ind,y_obj in enumerate(obj):
        y_obj.reverse()
        obj[ind] = y_obj
    return obj

def Fl2u(obj):#转换field方向（左到上）(finished
    temp = list(zip(obj[0],obj[1],obj[2],obj[3]))
    obj = []
    for each in temp:
        obj.extend([list(each)])
    return obj

def leftward(obj,score):#使field向左滑动(finished
    for ind,y_obj in enumerate(obj):
        num_zero = y_obj.count(0)
        for i in range(0,num_zero):
            obj[ind].remove(0)
            obj[ind].append(0)
        if y_obj[0] == y_obj[1]:
            obj[ind][0] = y_obj[0] + y_obj[1]
            score += obj[ind][0]
            obj[ind][1] = obj[ind][2]
            obj[ind][2] = obj[ind][3]
            obj[ind][3] = 0
        if y_obj[1] == y_obj[2]:
            obj[ind][1] = y_obj[1] +y_obj[2]
            score += obj[ind][1]
            obj[ind][2] = obj[ind][3]
            obj[ind][3] = 0
        if y_obj[2] == y_obj[3]:
            obj[ind][2] = y_obj[2] + y_obj[3]
            score += obj[ind][2]
            obj[ind][3] = 0
    return obj,score

def move(ope,nfield,score):
    if ope == 3:
        nfield,score = leftward(nfield,score)
    elif ope == 0:
        nfield = Fl2u(nfield)
        nfield,score = leftward(nfield,score)
        nfield = Fl2u(nfield)
    elif ope == 1:
        nfield = Fl2r(nfield)
        nfield,score = leftward(nfield,score)
        nfield = Fl2r(nfield)
    elif ope == 2:
        nfield = Fl2u(nfield)
        nfield = Fl2r(nfield)
        nfield,score = leftward(nfield,score)
        nfield = Fl2r(nfield)
        nfield = Fl2u(nfield)
    return nfield,score

def check(field):
    
    num = 0
    for each in field:
        for x in each:
            if x == 0:
                num += 1
    if num != 0:
        return [True,True]
    else:
        
        l =4
        nfield = copy.deepcopy(field)
        for i in range(4):
            res,n = move(i,field,0)
            if res != field:
                return [True,False]
            else:
                l -= 1
        
        if l == 0:
            return [False,False]

def game(ope,field,score):
    
    flag = check(field)
    if flag[0]:
        if flag[1]:
            
            field = add_num(field)
            return [True,field],score
        else:
            return [True,field],score
    else:
        return [False,field],score
    nfield = copy.deepcopy(field)
    field,score = move(ope,field,score)
    if field == nfield:
        return [True,field],score
    

if __name__ == '__main__':
    res,score = init()
    while True:
        if res[0]:
            print_screen(res[1],score)
            ope = in_ope(res[1])
            res,score = game(ope,res[1],score)
        else:
            print('lose')
            break