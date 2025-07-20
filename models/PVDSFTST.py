import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('nvl,vw->nwl',(x,A))
        return x.contiguous()

class GCN(nn.Module):
    def __init__(self,enc_in,c_in,c_out,d_model,dropout=0.1,order=2):
        super(GCN,self).__init__()
        self.nconv = nconv()
        c_in = (order+1)*c_in
        self.mlp = nn.Linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

        self.variable1 = torch.randn(enc_in, d_model, requires_grad=True)
        self.variable2 = torch.randn(enc_in, d_model, requires_grad=True)
        self.alpha =  1
        self.lin1 = nn.Linear(d_model,d_model)
        self.lin2 = nn.Linear(d_model,d_model)
    def forward(self,x):
        # x [B M d_model]
        out = [x]
        nodevec1 = torch.tanh(self.alpha*self.lin1(self.variable1.to(x.device)))
        nodevec2 = torch.tanh(self.alpha*self.lin2(self.variable2.to(x.device)))
        a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
        a_adjust = F.relu(torch.tanh(self.alpha*a))

        x1 = self.nconv(x,a_adjust)
        out.append(x1)
        for k in range(2, self.order + 1):
            x2 = self.nconv(x1,a_adjust)
            out.append(x2)
            x1 = x2
        h = torch.cat(out,dim=-1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training) # B M d_model
        return h


class TransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads)
        self.linear1 = nn.Linear(d_model, 4 * d_model)
        self.linear2 = nn.Linear(4 * d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x,x_external):
        # Self-attention
        attn_output, extract_attention = self.self_attn(x, x_external, x_external)
        x = x + attn_output
        x = self.norm1(x)
        
        # Feed-forward network
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        x = x + attn_output
        x = self.norm2(x)
        
        return x

class Extract_Variable_Dependecnce(nn.Module):
    def __init__(self, d_model,num_layers = 2, num_heads=4):
        super().__init__()
        #self.transformer = TransformerLayer(d_model, num_heads=4)
        self.layers = nn.ModuleList([TransformerLayer(d_model, num_heads=num_heads) for _ in range(num_layers)])
    
    def forward(self, x_target,x_external ):
        '''
        x_target: B 1  L d_model
        x_external: B M-1 L d_model
        out: B L d_model
        '''
        B,M,L,C = x_external.shape
        B,_,L,C = x_target.shape
        x_external = x_external.permute(0,2,1,3).reshape(B*L,M,C)
        x_target = x_target.permute(0,2,1,3).reshape(B*L,_,C)
        
        x_external = x_external.permute(1, 0, 2)
        x_target = x_target.permute(1, 0, 2)
        for layer in self.layers:
            x_target = layer(x_target,x_external)
        output = x_target.permute(1, 0, 2) # B*L 1 C
        output = output.reshape(B,L,C)
        return output
class LSTMModel(nn.Module):
    def __init__(self,patch_num, input_size, d_model, num_layers=2):
        super(LSTMModel, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, d_model, num_layers, batch_first=True)
        self.map = nn.Linear(d_model*patch_num,d_model)
    def forward(self, x):
        '''
        x: B M patch_num patch_len
        out: B M d_model
        '''
        # reshape
        B,M,L,C =x.shape
        x = x.reshape(B*M,L,C) # B M patch_num patch_len -> B*M patch_num patch_len
        h0 = torch.zeros(self.num_layers, x.size(0), self.d_model).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.d_model).to(x.device)
        out, _ = self.lstm(x, (h0, c0)) # B M patch_num d_model
        out = out.reshape(B,M,L,self.d_model) 
        return out

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.patch_len = configs.patch_len
        self.patch_num = int(self.seq_len/self.patch_len)
        self.d_model = configs.d_model
        self.num_layers = configs.e_layers
        self.num_heads = configs.n_heads
        self.enc_in = configs.enc_in
        # 1.1 lstm时序特征提取
        self.lstm = LSTMModel(patch_num=self.patch_num,input_size=self.patch_len,d_model=self.d_model)
        # 1.2 目标变量对外部变量的关系提取
        self.extract_variable_dependence = Extract_Variable_Dependecnce(d_model=self.d_model,num_layers=self.num_layers,num_heads=self.num_heads)
        # 2.1 linear编码
        self.linear = nn.Linear(self.seq_len,self.d_model)
        # 2.2 图神经网络
        self.gnn = GCN(enc_in=self.enc_in,c_in=self.d_model,c_out=self.d_model,d_model=self.d_model,dropout=0.1,order=2)
        # 3 attention 多段融合
        self.cross_attn = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=self.num_heads)
        # 4 预测
        self.fc = nn.Linear(self.seq_len+self.d_model,self.pred_len)
    def forward(self, input):       
        '''
        input: [Batch, Sequence Length, Multi-Variables]
        out: [Batch, Sequence Length, Single-Variable]
        '''
        
        B, L, M = input.shape
        # 1.0 对x进行patch
        x = input.permute(0,2,1) # [B, L, M] -> [B, M, L]
        x = x.reshape(B,M,self.patch_num,self.patch_len) 
        # 1.1 使用共享的 LSTM 对 内部变量 和 外部变量 做时序特征提取
        lstm_out = self.lstm(x) # [B,M,L',d_model]
        x_external = lstm_out[:,:-1,:,:] 
        x_target = lstm_out[:,-1:,:,:] 
        # 1.2 使用注意力机制，内部变量作为query，外部变量作为key和value
        patch_out = self.extract_variable_dependence(x_target,x_external) # [Batch, L', d_model]
        # print('patch_out shape: [Batch, L, d_model]', patch_out.shape)

        # 2.1 使用Linear对变量进行编码
        x = input.permute(0,2,1) # B M L
        x = self.linear(x) # B M d_model
        # 2.2 gnn交互，并取出最后一维目标变量
        gnn_out = self.gnn(x)[:,-1:,:] # B M d_model 取target变量： B 1 d_model
        # 3 cross attention 逐段学习
        fusion_out, integration_attention = self.cross_attn(gnn_out.permute(1,0,2), patch_out.permute(1,0,2),patch_out.permute(1,0,2))
       
        # 4. 为了调效果，把目标变量的历史加上表征，做了一个残差然后线性层映射
        out = fusion_out.permute(1,0,2) # -> B 1 d_model 
        out = self.fc(torch.cat([torch.tanh(out),input[:,:,-1:].permute(0,2,1)],dim=-1)).permute(0,2,1) # [Batch, T, 1]
        return out 
