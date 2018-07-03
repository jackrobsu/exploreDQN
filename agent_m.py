#-*- coding:utf-8 -*-
import sys , getopt
import numpy as np
from world import *
import json
import matplotlib as plt
from matplotlib.ticker import MultipleLocator , FormatStrFormatter
import pylab as pl
import ConfigParser
from other import OtherDQN

class Agent(object):
    def __init__(self,Config,recode=False,initializePosition=None):
        self.samples = []
        self.transitions = []
        self.unknownlabel = []
        self.mem = []
        # 初始化世界
        self.world = World(Config.get("parameters",Config.get("parameters","mapparameter")))
        # agent的状态数
        self.states = self.world.stateNumber

        self.policy = None
        # 是否把需要把模型写入文件
        self.recode = recode
        # agent初始位置
        self.initializePosition = initializePosition

        # agent的执行动作数，如果动作数是5,说明agent可以采取的行为是上、下、左、右、原地不动，
        # 如果动作数是4，说明agent可以采取的行为包括上、下、左、右。
        self.action_dim = 5
        # self.askAction_dim = self.action_dim + 1
        # agent的询问动作数（目前只设置了一个）
        self.askAction_dim = 1

        self.queryAction = None
        self.extra_reward = None
        self.extra_action = None

        # 是非类型的问题的代价（如问”应该往前走吗？“）
        self.yesnoQueryReward = -3
        # 一般类型的问题的代价（如问”现在怎么走？“）
        self.askQueryReward = -1.5

        self.asktimes = []
        self.allAskTimes = []

    def increaseReward(self):
        '''
            用于对整个执行序列的reward进行调整（目前好像没有用到，如果需要，可以修改）
        '''
        def cursum(mem):
            sum = float(0)
            rate = 0.9
            for i in mem :
                sum += i * rate
                rate = rate * rate
                if rate < 0.00001 :
                    rate = 0
            return sum
        if not self.mem :
            return
        flag = False
        # flag = True
        if flag :
            rs = [ r[3] for r in self.mem ]
            
            if self.mem and self.mem[-1][3] == self.world.targetReward :   
                rs = []
                self.mem.reverse()
                rate = 0.9
                Reward = self.mem[0][3]
                g = self.mem[0]
                # self.samples[g[0]][g[1]].append(g[2:4])
                rs.append(g[3])
                for r in self.mem[1:] :
                    r[3] = max(0.1,Reward * rate)
                    rate = rate * rate
                    rs.append(r[3])
                    # self.samples[r[0]][r[1]].append(r[2:4])
                rs.reverse()
                # self.mem.reverse()
                print(rs)
                # exit(0)
            temp = []
            # rs = np.cumsum(rs)
            # br = rs[-1]
            for i , r in enumerate(self.mem) :
                # if i == 0 :
                #     r[3] = br
                # else:
                #     r[3] = br - rs[i-1]
                # r[3] = cursum(rs[i+1:])
                temp.append(r[3])
                self.samples[r[0]][r[1]].append(r[2:4])
            return
        else:
            if self.mem and self.mem[-1][3] != self.world.targetReward :
                # rs = [ r[3] for r in self.mem ]
                # rs = np.cumsum(rs)
                # br = rs[-1]
                for i , r in enumerate(self.mem) :
                    # if i == 0 :
                    #     r[3] = br
                    # else:
                    #     r[3] = br - rs[i-1]
                    self.samples[r[0]][r[1]].append(r[2:4])
                return

            # for r in self.mem :
            #     self.samples[r[0]][r[1]].append(r[2:4])
            # return

            self.mem.reverse()
            rate = 0.9
            Reward = self.mem[0][3]
            g = self.mem[0]
            self.samples[g[0]][g[1]].append(g[2:4])
            for r in self.mem[1:] :
                r[3] = max(0.1,Reward * rate)
                rate = rate * rate
                self.samples[r[0]][r[1]].append(r[2:4])
            
        

    def executeAction(self,state):
        '''
            动作执行函数，会把动作发送给世界，同时收到执行后的结果（目前没用到）
        '''
        r = self.world.simulation(state,self.policy[state])
        self.mem.append(r)
        # self.samples[r[0]][r[1]].append(r[2:4])
        return r[2:5]

    def execute_Action(self,action,state,stayON=True):
        '''
            动作执行函数，会把动作发送给世界，同时收到执行后的结果
        '''
        r = self.world.simulation(state,action,stayON=True)
        # self.mem.append(r)
        # self.samples[r[0]][r[1]].append(r[2:4])
        return r[2:5]

    def getTarget(self,state):
        '''
            用于判断当前状态是否是目标状态
        '''
        return self.world.isTarget(state)

    def getInitPosition(self):
        '''
            用于获取agent初始位置
        '''
        if self.initializePosition == None :
            p = np.random.randint(0,len(agent.world.legalstate))
            # p = self.initializePosition 
            p = self.world.legalstate[p]
            self.initializePosition = p
            self.world.oP = p
            self.world.Print(self.world.state_to_map[p],True,isshow=True)
            return p
        else:
            p = self.initializePosition
            self.world.oP = p            
            self.world.Print(self.world.state_to_map[p],True)            
            return self.initializePosition
        with open("p.txt",'w') as out :
            p = np.random.randint(0,len(agent.world.legalstate))
            p = self.initializePosition 
            out.write(str(p))
        return self.world.legalstate[p]

    def askGuider(self,state,stayON=True):
        '''
            询问Guider现在该往哪里走
            返回的是一个列表，包含可以走的动作（可以多个）
            stayON这个参数表示，agent的动作是否包含“待在原地”这个行为，为真表示包含该行为。
        '''
        return self.world.SuggestionForNextActions(state,stayON)

    def actionMapping(self,action,state,stayON=True):
        '''
            动作映射函数
            action参数是当前agent的动作
            如果该动作是执行动作，则直接返回该动作
            如果该动作是询问动作，则会先询问，然后返回被建议的动作
        '''
        if action < self.action_dim :
            self.asktimes.append(0)
            return action
        elif action < self.action_dim + self.askAction_dim - 1 :
            # if not guessed, the wrong action should be penal.
            # if guessed, the right action should be executed and give a positive reward
            actionset = self.askGuider(state,stayON)
            exec_action = action - self.action_dim
            if exec_action in actionset :
                self.extra_reward = 1
                self.extra_action = exec_action
                self.queryAction = action
                return exec_action
            else:
                #whether use a stay action or control the flow in the outer
                self.extra_reward = -1
                self.extra_action = exec_action
                self.queryAction = action
                return exec_action
        else:
            actionset = self.askGuider(state,stayON)
            self.asktimes.append(1)
            exec_action = np.random.choice(actionset)
            self.extra_reward = 1
            self.extra_action = exec_action
            self.queryAction = action
            return exec_action

    def addExtraMemory(self,dqn,memory):
        '''
            添加动作样本
        '''
        def Clear():
            self.queryAction = None
            self.extra_action = None
            self.extra_reward = None
        
        over = False
        # dqn.addSwitchMemory([memory[0],[1.0,0.0]])
     
        if self.queryAction is None :
            dqn.addMemory(memory)            
            Clear()
            return over
        if self.extra_reward > 0 :
            # [state_,a,r,nextstate_,done]
            execmemory = list(memory)
            execmemory[2] += self.extra_reward
            dqn.addMemory(execmemory)
            querymemory = list(memory)
            querymemory[1] = self.queryAction
            querymemory[2] += self.extra_reward 
            if self.queryAction == self.action_dim + self.askAction_dim - 1 :
                querymemory[2] += self.askQueryReward
            else:
                querymemory[2] += self.yesnoQueryReward

            dqn.addMemory(querymemory)
            # for a in range(self.action_dim) :
            #     if a == memory[1] :
            #         continue
            #     m = list(memory)
            #     m[1] = a
            #     m[2] = -1.5
            #     m[-1] = True
            #     dqn.addMemory(m)
            dqn.addSwitchMemory([memory[0],[1.0,0.0]])
            over = False
        else:
            memory[-1] = True
            execmemory = list(memory)
            execmemory[2] += self.extra_reward
            dqn.addMemory(execmemory)
            querymemory = list(memory)
            querymemory[1] = self.queryAction
            querymemory[2] = self.extra_reward 
            if self.queryAction == self.action_dim + self.askAction_dim - 1 :
                querymemory[2] += self.askQueryReward
            else:
                querymemory[2] += self.yesnoQueryReward
            dqn.addMemory(querymemory)

            for a in range(self.action_dim) :
                if a == memory[1] :
                    continue
                m = list(memory)
                m[1] = a
                m[2] = 0.5
                m[-1] = True
                dqn.addMemory(m)
            dqn.addSwitchMemory([memory[0],[0,1.0]])

            over = False
        
        Clear()
        return over
      

if __name__ == '__main__' :
    opt , args = getopt.getopt(sys.argv[1:],"rup:sf:")
    # 是否需要将模型写入到文件
    recode = False
    # 用于指定初始位置
    position = None
    # 用于在终端展示地图的标记
    showflag = False
    # 需要被保存的图片的文件名
    figurefile = "figure"

    for op , value in opt :
        if op == "-r" :
            recode = True
        elif op == "-p" :
            position = int(value)
        elif op == "-s" :
            showflag = True
        elif op == "-f" :
            figurefile = value

    Config = ConfigParser.ConfigParser()
    Config.read("config.config")
    section = "parameters"
    
    agent = Agent(Config,recode=recode,initializePosition=position)

    successrate1 = None
    successrate2 = None
    avg_rewards_1 = None
    avg_rewards_2 = None

    odqn_query = OtherDQN(Query=True)
    odqn_noquery = OtherDQN(Query=False)
    

    stayON = True

    for itype in range(2) :

        state = agent.getInitPosition()

        successtimes = 0
        times = 0
        steps = []
        rewards = []
        maxtimes = Config.getint(section,"maxtimes")
        maxsteps = Config.getint(section,"maxsteps")
        avg_rewards = []
        avg_all_rewards = []
        ep = Config.getint(section,"ep")
        if not showflag :
            agent.world.showFlag = False
        # agent.world.showFlag = False
        
        for I in range(maxtimes):
            reward = 0.0
            count = 0
            flag = False
            for i in range(maxsteps):
                # nextstate ,r ,c = agent.executeAction(state)
                state_ = agent.world.state_to_coordinate(state) 
                if itype == 1 :           
                    a , israndom = odqn_noquery.egreedy_action(state_)
                    # agent.actionMapping(a,state_,stayON)

                else:
                    a , israndom = odqn_query.egreedy_action(state_)
                    a = agent.actionMapping(a,state_,stayON)
                    if israndom :
                        agent.asktimes[-1] = 0
                
                nextstate , r , c = agent.execute_Action(a,state,stayON)
                nextstate_ = agent.world.state_to_coordinate(nextstate)
                reward += r
                done = False
                if c == -1 :
                    done = True
                elif c == 0 :
                    done = False
                if itype == 1 :
                    odqn_noquery.addMemory([state_,a,r,nextstate_,done])
                    
                    # agent.addExtraMemory(odqn,[state_,a,r,nextstate_,done])
                else:
                    # odqn_query.perceive(state_,a,r,nextstate_,done)
                    over = agent.addExtraMemory(odqn_query,[state_,a,r,nextstate_,done])
                    if over :
                        c = -1
                    
                if c == -1 :
                    if agent.getTarget(nextstate) :
                        print('reward ',reward)
                        print('step ',count)
                        steps.append(count)
                        rewards.append(reward)
                        successtimes += 1
                        flag = True
                    break
                else:
                    state = nextstate
                    count += 1
            if flag == False :
                steps.append(count)
                rewards.append(reward)
            # if itype == 0 :
            #     agent.allAskTimes.append(np.sum(agent.asktimes))
            #     agent.asktimes = []

            state = agent.getInitPosition()
            # if I > maxtimes / 2 :
            #     agent.world.showFlag = showflag
            #     if I % 10 == 0 :
            #         agent.world.showFlag = True 
            #     else:
            #         agent.world.showFlag = False
            # else:
            #     agent.world.showFlag = False  
            
                          
            times += 1
            if times % ep == 0 :
                avg_rewards.append(np.mean(rewards,axis=0))
                avg_all_rewards.append([np.mean(rewards,axis=0),times])
                rewards = []

        print('success rate %d / %d = %.10f' % (successtimes, times,float(successtimes)/float(times)))
        if itype == 0 :
            successrate1 = float(successtimes)/float(times)
            avg_rewards_1 = list(avg_rewards)
   
        elif itype == 1 :
            successrate2 = float(successtimes)/float(times)
            avg_rewards_2 = list(avg_rewards)

    print('success rate %.10f' % successrate1)
    print('success rate %.10f' % successrate2)
    
    
    xmajorLocator   = MultipleLocator(Config.getint(section,"xmajorLocator")) #将x主刻度标签设置为20的倍数  
    xmajorFormatter = FormatStrFormatter('%d') #设置x轴标签文本的格式  
    xminorLocator   = MultipleLocator(Config.getint(section,"xminorLocator")) #将x轴次刻度标签设置为5的倍数  
    
    ymajorLocator   = MultipleLocator(Config.getfloat(section,"ymajorLocator")) #将y轴主刻度标签设置为0.5的倍数  
    ymajorFormatter = FormatStrFormatter('%1.1f') #设置y轴标签文本的格式  
    yminorLocator   = MultipleLocator(Config.getfloat(section,"yminorLocator")) #将此y轴次刻度标签设置为0.1的倍数  
    ax = pl.subplot(111)

    plot1 = pl.plot([i for i in range(maxtimes/ep)],avg_rewards_1,'og-',label="test1")
    plot2 = pl.plot([i for i in range(maxtimes/ep)],avg_rewards_2,'ob-',label="test2")
    
    pl.title("{}_maxtimes_{}_maxsteps_{}_S_{}".format(Config.get(section,Config.get(section,"figureparmeter")),Config.getint(section,"maxtimes"),Config.getint(section,"maxsteps"),position))
    pl.xlabel("runs/{}episodes".format(ep))
    pl.ylabel("avg_reward")
    
    #设置主刻度标签的位置,标签文本的格式  
    ax.xaxis.set_major_locator(xmajorLocator)  
    ax.xaxis.set_major_formatter(xmajorFormatter)  
    
    ax.yaxis.set_major_locator(ymajorLocator)  
    ax.yaxis.set_major_formatter(ymajorFormatter)  
    
    #显示次刻度标签的位置,没有标签文本  
    ax.xaxis.set_minor_locator(xminorLocator)  
    ax.yaxis.set_minor_locator(yminorLocator)  
    
    ax.xaxis.grid(True, which='major') #x坐标轴的网格使用主刻度  
    ax.yaxis.grid(True, which='minor') #y坐标轴的网格使用次刻度 
    # pl.plot([i for i in range(maxtimes)],steps)
    pl.legend()
    pl.savefig(figurefile)    
    pl.show()
    # with open("rewards.txt",'w') as f :
    #     f.write("{}\n".format(avg_rewards))
    #     f.write("{}".format(avg_all_rewards))
        