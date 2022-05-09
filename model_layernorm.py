import math

import torch
from torch import nn, sigmoid, Tensor
from torch.nn import TransformerEncoderLayer, MultiheadAttention, TransformerEncoder
from torch.nn.modules.transformer import _get_activation_fn
from typing import Optional, Any
from torch import functional as F


def GLU(input_tensor):
	tensor_a, tensor_b = torch.split(input_tensor, 128, 1)
	return tensor_a * sigmoid(tensor_b)


class ResBlock(nn.Module):

    def __init__(self, filter_shape):
        super(ResBlock,self).__init__()

        self.kernel_size = filter_shape
        #self.layernorm_dim = layernorm_dim
        self.layernorm = nn.LayerNorm(256)

        self.conv = nn.Conv2d(128,256,self.kernel_size,1,1)

    def forward(self,x):
        out = x.permute(0,2,3,1).contiguous()
        out = self.layernorm(out)
        out = out.permute(0,3,1,2).contiguous()
        out = GLU(out)
        out = self.conv(out)
        return x + out


class PlainBlock(nn.Module):

    def __init__(self, filter_shape):
        super(PlainBlock, self).__init__()

        self.kernel_size = filter_shape
        # self.layernorm_dim = layernorm_dim
        self.layernorm = nn.LayerNorm(256)

        self.conv = nn.Conv2d(128, 256, self.kernel_size, 1, 1)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        out = self.layernorm(x)
        out = out.permute(0, 3, 1, 2).contiguous()
        out = GLU(out)
        out = self.conv(out)
        return out

class NormBlock(nn.Module):

    def __init__(self):
        super(NormBlock, self).__init__()

        #self.kernel_size = filter_shape
        # self.layernorm_dim = layernorm_dim
        self.layernorm = nn.LayerNorm(256)

        #self.conv = nn.Conv2d(128, 256, self.kernel_size, 1, 1)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        out = self.layernorm(x)
        out = out.permute(0, 3, 1, 2).contiguous()
        out = GLU(out)

        #out = self.conv(out)
        return out

class TransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

      
def _init_weights(m):
    """
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)
        
        
class stdDCS_SI(nn.Module):

    def __init__(self, proj = 0.3, batch_size = 1,  channel = 256, N_block = 4, kernel_size = 3):
        super(stdDCS_SI,self).__init__()

        self.batch_size = batch_size
        self.channel = channel
        self.N_block = N_block
        self.kernel_size = kernel_size
        self.proj = proj
        #self.seq_len = seq_len
        #self.layernorm_dim = [self.channel,self.seq_len, 1]

        #位置编码
        #self.pos_encoder = PositionalEncoding(23, 0.1)

        #transformer-encoder
        encoder_layers = TransformerEncoderLayer(26, 2, 128, 0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)

        #初始化卷积
        self.conv_1 = nn.Conv2d(1,256,(self.kernel_size,28),1,1)

        #残差块
        self.ResBlock = ResBlock(self.kernel_size)

        #平铺块
        self.PlainBlock = PlainBlock(self.kernel_size)

        #标准块
        self.NormBlock = NormBlock()

        #解码器部分
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(self.proj)
        self.conv_fc0 = nn.Conv2d(128,256,(1,1),1,0)
        self.conv_fc1 = nn.Conv2d(256,2,(1,1),1,0)
        self.apply(_init_weights)

    def forward(self,x):
        #转换输入的维度
        x = torch.squeeze(x)

        x = torch.unsqueeze(x,1)

        #x = self.pos_encoder(x)

        #Transformer编码器
        x = self.transformer_encoder(x)

        #变换维度
        x = torch.unsqueeze(x,0)
        x = x.permute(0,2,1,3)

        #初始化卷积
        out = self.conv_1(x)

        #stage1
        for i in range(self.N_block):
            out = self.ResBlock(out)

        out = self.PlainBlock(out)

        #stage2
        for i in range(self.N_block):
            out = self.ResBlock(out)

        out = self.NormBlock(out)

        #解码器
        out = self.conv_fc0(out)
        out = self.relu(out)
        out = self.drop(out)
        out = self.conv_fc1(out)
        out = self.relu(out)
        out = torch.squeeze(out,-1)
        #out = out.permute(0,2,1)

        return out


if __name__ == '__main__':
    model = stdDCS_SI()
    print(model)
