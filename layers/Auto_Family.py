import time
import torch
import torch.nn as nn
from torch.nn.functional import interpolate
import math


def decor_time(func):
    def func2(*args, **kw):
        now = time.time()
        y = func(*args, **kw)
        t = time.time() - now
        print('call <{}>, time={}'.format(func.__name__, t))
        return y

    return func2


class AutoCorrelation(nn.Module):
    """
    AutoCorrelation Mechanism with the following two phases:
    (1) period-based dependencies discovery
    (2) time delay aggregation
    This block can replace the self-attention family mechanism seamlessly.
    """

    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1, output_attention=False,
                 configs=None, device=None):
        super(AutoCorrelation, self).__init__()
        print('Autocorrelation used !')
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.agg = None
        self.use_wavelet = 0  # configs.wavelet
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # @decor_time
    def time_delay_agg_training(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the training phase.
        """
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            pattern = torch.roll(tmp_values, -int(index[i]), -1)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg  # size=[B, H, d, S]

    def time_delay_agg_inference(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the inference phase.
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = torch.arange(length, device=self.device).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1)
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        weights = torch.topk(mean_value, top_k, dim=-1)[0]
        delay = torch.topk(mean_value, top_k, dim=-1)[1]
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
            tmp_delay.to(self.device)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg

    def time_delay_agg_full(self, values, corr):
        """
        Standard version of Autocorrelation
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = torch.arange(length, device=self.device).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1)
        # find top k
        top_k = int(self.factor * math.log(length))
        weights = torch.topk(corr, top_k, dim=-1)[0]
        delay = torch.topk(corr, top_k, dim=-1)[1]
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[..., i].unsqueeze(-1)
            tmp_delay.to(self.device)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * (tmp_corr[..., i].unsqueeze(-1))
        return delays_agg

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        if L > S:
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        # period-based dependencies
        if self.use_wavelet != 2:
            if self.use_wavelet == 1:
                j_list = self.j_list
                queries = queries.reshape([B, L, -1])
                keys = keys.reshape([B, L, -1])
                Ql, Qh_list = self.dwt1d(queries.transpose(1, 2))  # [B, H*D, L]
                Kl, Kh_list = self.dwt1d(keys.transpose(1, 2))
                qs = [queries.transpose(1, 2)] + Qh_list + [Ql]  # [B, H*D, L]
                ks = [keys.transpose(1, 2)] + Kh_list + [Kl]
                q_list = []
                k_list = []
                for q, k, j in zip(qs, ks, j_list):
                    q_list += [interpolate(q, scale_factor=j, mode='linear')[:, :, -L:]]
                    k_list += [interpolate(k, scale_factor=j, mode='linear')[:, :, -L:]]
                queries = torch.stack([i.reshape([B, H, E, L]) for i in q_list], dim=3).reshape([B, H, -1, L]).permute(
                    0, 3, 1, 2)
                keys = torch.stack([i.reshape([B, H, E, L]) for i in k_list], dim=3).reshape([B, H, -1, L]).permute(0,
                                                                                                                    3,
                                                                                                                    1,
                                                                                                                    2)
            else:
                pass
            q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)  # size=[B, H, E, L]
            k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
            res = q_fft * torch.conj(k_fft)
            corr = torch.fft.irfft(res, dim=-1)  # size=[B, H, E, L]

            # time delay agg
            if self.training:
                V = self.time_delay_agg_training(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1,
                                                                                                        2)  # [B, L, H, E], [B, H, E, L] -> [B, L, H, E]
            else:
                V = self.time_delay_agg_inference(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
        else:
            V_list = []
            queries = queries.reshape([B, L, -1])
            keys = keys.reshape([B, L, -1])
            values = values.reshape([B, L, -1])
            Ql, Qh_list = self.dwt1d(queries.transpose(1, 2))  # [B, H*D, L]
            Kl, Kh_list = self.dwt1d(keys.transpose(1, 2))
            Vl, Vh_list = self.dwt1d(values.transpose(1, 2))
            qs = Qh_list + [Ql]  # [B, H*D, L]
            ks = Kh_list + [Kl]
            vs = Vh_list + [Vl]
            for q, k, v in zip(qs, ks, vs):
                q = q.reshape([B, H, E, -1])
                k = k.reshape([B, H, E, -1])
                v = v.reshape([B, H, E, -1]).permute(0, 3, 1, 2)
                q_fft = torch.fft.rfft(q.contiguous(), dim=-1)
                k_fft = torch.fft.rfft(k.contiguous(), dim=-1)
                res = q_fft * torch.conj(k_fft)
                corr = torch.fft.irfft(res, dim=-1)  # [B, H, E, L]
                if self.training:
                    V = self.time_delay_agg_training(v.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
                else:
                    V = self.time_delay_agg_inference(v.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
                V_list += [V]
            Vl = V_list[-1].reshape([B, -1, H * E]).transpose(1, 2)
            Vh_list = [i.reshape([B, -1, H * E]).transpose(1, 2) for i in V_list[:-1]]
            V = self.dwt1div((Vl, Vh_list)).reshape([B, H, E, -1]).permute(0, 3, 1, 2)
            # corr = self.dwt1div((V_list[-1], V_list[:-1]))

        if self.output_attention:
            return (V.contiguous(), corr.permute(0, 3, 1, 2))  # size = [B, L, H, E]
        else:
            return (V.contiguous(), None)


class AutoCorrelationLayer(nn.Module):
    def __init__(self, correlation, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AutoCorrelationLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_correlation(
            queries,
            keys,
            values,
            attn_mask
        )

        out = out.view(B, L, -1)
        return self.out_projection(out), attn


# Auxiliary part for Autoformer Encoder and Decoder

class my_Layernorm(nn.Module):
    """
    Special designed layernorm for the seasonal part
    """
    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        # print('kernel_size:', self.kernel_size)
        if not isinstance(self.kernel_size, int):
            self.kernel_size = self.kernel_size[0]
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1-math.floor((self.kernel_size - 1) // 2), 1)
        end = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1) // 2), 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class series_decomp_multi(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]
        self.layer = torch.nn.Linear(1, len(kernel_size))

    def forward(self, x):
        moving_mean=[]
        for func in self.moving_avg:
            moving_avg = func(x)
            moving_mean.append(moving_avg.unsqueeze(-1))
        moving_mean=torch.cat(moving_mean,dim=-1)
        moving_mean = torch.sum(moving_mean*nn.Softmax(-1)(self.layer(x.unsqueeze(-1))),dim=-1)
        res = x - moving_mean
        return res, moving_mean


class FourierDecomp(nn.Module):
    def __init__(self):
        super(FourierDecomp, self).__init__()
        pass

    def forward(self, x):
        x_ft = torch.fft.rfft(x, dim=-1)
