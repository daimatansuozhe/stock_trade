

from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

########注意力模块############
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.view(B, H // self.window_size, self.window_size,
                   W // self.window_size, self.window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(-1, self.window_size * self.window_size, C)  # (B*num_windows, N, C)

        qkv = self.qkv(x).reshape(x.shape[0], x.shape[1], 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # (3, B*, num_heads, N, dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(x.shape[0], x.shape[1], C)

        out = self.proj(out)
        out = out.view(B, H // self.window_size, W // self.window_size,
                       self.window_size, self.window_size, C)
        out = out.permute(0, 1, 3, 2, 4, 5).contiguous()
        out = out.view(B, H, W, C)
        return out


class MultiScaleWindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_sizes=[2, 4]):
        super(MultiScaleWindowAttention, self).__init__()
        self.layers = nn.ModuleList([
            WindowAttention(dim, w_size, num_heads) for w_size in window_sizes
        ])

    def forward(self, x):
        # x: [B, C, H, W] -> [B, H, W, C]
        x = x.permute(0, 2, 3, 1).contiguous()
        outs = []
        for attn in self.layers:
            outs.append(attn(x))
        out = sum(outs) / len(outs)  # mean fusion
        out = out.permute(0, 3, 1, 2).contiguous()  # -> [B, C, H, W]
        return out

#dim为特征维度，修改为对应即可
msa = MultiScaleWindowAttention(dim=768, num_heads=4)

######################################


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.bert = BertModel(config, add_pooling_layer=True)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        #此处插入attention模块

        sequence_output = outputs[0]  # [B, seq_len, hidden_size]
        B, L, C = sequence_output.shape
        # 1. 计算 padding 后的合法长度 L_pad，使得 sqrt(L_pad) 是整数并能用于 window_size
        def get_valid_square_len(L, window_sizes=[2, 4]):
            import math
            def is_valid(n):
                sqrt_n = int(math.sqrt(n))
                return sqrt_n * sqrt_n == n and all(sqrt_n % w == 0 for w in window_sizes)

            L_pad = L
            while not is_valid(L_pad):
                L_pad += 1
            return L_pad

        L_pad = get_valid_square_len(L)
        pad_len = L_pad - L
        H = int(L_pad ** 0.5)

        # 2. 补零到合法长度
        if pad_len > 0:
            pad_tensor = torch.zeros((B, pad_len, C), device=sequence_output.device)
            sequence_output = torch.cat([sequence_output, pad_tensor], dim=1)  # [B, L_pad, C]


        x = sequence_output.transpose(1, 2).reshape(B, C, H, H)

        x = msa(x)  # 多尺度注意力模块
        x = x.reshape(B, C, -1).transpose(1, 2)  # [B, L, C]
        pooled_output = x[:, 0, :]  # 使用 [CLS] token

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()  # 回归任务
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()  # 分类任务
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
