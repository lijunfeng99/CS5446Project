

class RewardAgent():
    def __init__(self, model_name= "reward_agent"):
        self.name = model_name
    
    def load(self):
        pass

    def action2index(self, action, board):
        idx = action
        while (idx < 42) and (board[idx] == 0):
            idx += 7
        return idx - 7

    def get_reward(self, board,idx,rows=6,cols=7,label=1):
        # for i in range(rows):
        #     for j in range(cols):
        #         print(int(board[i*cols+j]),end=" ")
        #     print()

        total_reward=0
        rewards=[-0.03, 0.1, 0.8, 6.4]#目前的规则,如果连成4颗棋子,得到12分,如果3颗棋子得到6分,如果2颗棋子2分,1颗棋子0分。
        #反正是初学者也不用考虑那么多,先把代码框架完成。
        col_id = idx%cols#行号和列号确定下来
        row_id = (idx - col_id)//cols
        # print(f"idx:{idx}, row_id:{row_id}, col_id:{col_id}")
        #如果这个位置下面没有棋子,那么这个位置不能摆棋
        if (row_id<rows-1) and (board[(row_id+1)*cols+col_id]==0):
            return -1
        #对于8个方位检查
        for position in [[row_id-1,col_id-1],[row_id-1,col_id],[row_id-1,col_id+1],[row_id,col_id-1],               
            # [row_id,col_id+1],   [row_id+1,col_id-1],[row_id+1,col_id],[row_id+1,col_id+1]
            ]:
            #如果是己方的棋子就要考虑到底有几颗,目前已经确定的有2颗
            #row_id-cur_row,col_id-cur_col 这个方向还有反方向
            cur_row,cur_col=row_id,col_id
            direction=[row_id-position[0],col_id-position[1]]
            point_count=1#棋子的个数
            #只要没越界并且往前走还是己方棋子
            while (0<=cur_row-direction[0]<rows) and (0<=cur_col-direction[1]<cols) and \
                (board[(cur_row-direction[0])*cols+(cur_col-direction[1])]==label):
                    point_count+=1
                    #继续向前走
                    cur_row=cur_row-direction[0]
                    cur_col=cur_col-direction[1]
            #往另一个方向

            cur_row,cur_col=row_id,col_id
            #只要没越界并且往前走还是己方棋子
            while (0<=cur_row+direction[0]<rows) and (0<=cur_col+direction[1]<cols) and\
                (board[(cur_row+direction[0])*cols+(cur_col+direction[1])]==label):
                    point_count+=1
                    #继续向前走
                    cur_row=cur_row+direction[0]
                    cur_col=cur_col+direction[1]
            #考虑每个方向的情况得到一个total_reward
            # print(f"direction:{direction},point_count:{point_count}")
            # print(f"position:{position},point_count:{point_count}")
            total_reward+=rewards[point_count-1]
        return total_reward
    
    def __call__(self, obsevation, configuration):
        state = obsevation['board']
        mark = obsevation['mark']
        my_action = None
        mx_reward = -1000
        for action in range(7):
            idx = self.action2index(action, state)
            if idx < 0:
                continue

            cur_reward = self.get_reward(state, idx, label=mark)
            if cur_reward > mx_reward:
                mx_reward = cur_reward
                my_action = action

        return my_action