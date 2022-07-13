import torch
from torch import nn
# 导入实现的module
from model.module.Img_module import ImgModule
from model.module.Text_module import TextModule


class MultimodalModel(nn.Module):
    def __init__(self, args):
        super(MultimodalModel, self).__init__()
        self.TextModule_ = TextModule(args)
        self.ImgModule_ = ImgModule(args)

        self.multihead_attn = nn.MultiheadAttention(embed_dim=1000, num_heads=2, batch_first=True)
        self.linear_text_k1 = nn.Linear(1000, 1000)
        self.linear_text_v1 = nn.Linear(1000, 1000)
        self.linear_img_k2 = nn.Linear(1000, 1000)
        self.linear_img_v2 = nn.Linear(1000, 1000)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=1000, nhead=2, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

        # 辅助分类器
        self.classifier_img = nn.Sequential(
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 200),
            nn.ReLU(),
            nn.Linear(200, 40),
            nn.ReLU(),
            nn.Linear(40, 3),
        )
        # 辅助分类器
        self.classifier_text = nn.Sequential(
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 200),
            nn.ReLU(),
            nn.Linear(200, 40),
            nn.ReLU(),
            nn.Linear(40, 3),
        )
        # 主分类器
        self.classifier_multi = nn.Sequential(
            # nn.Flatten(start_dim=2, end_dim=- 1),
            nn.Flatten(),
            nn.Linear(2000, 1000),
            nn.ReLU(),
            nn.Dropout(p=args.dropout),
            nn.Linear(1000, 200),
            nn.ReLU(),
            nn.Linear(200, 40),
            nn.ReLU(),
            nn.Linear(40, 3),
        )

    def forward(self, bach_text=None, bach_img=None):
        if bach_text is None and bach_img is None:
            # print('No input')
            return None, None, None

        # 单模态数据
        if bach_text is not None and bach_img is None:
            # print('only input text')
            _, text_out = self.TextModule_(bach_text)
            text_out = self.classifier_text(text_out)
            return text_out, None, None

        if bach_text is None and bach_img is not None:
            # print('only input image')
            img_out = self.ImgModule_(bach_img)
            img_out = self.classifier_img(img_out)
            return None, img_out, None

        # 多模态数据
        _, text_out = self.TextModule_(bach_text)  # N, E
        img_out = self.ImgModule_(bach_img)  # N, E

        # 融合策略1：向量拼接
        # multi_out = torch.cat((text_out, img_out), 1)

        # 融合策略2：使用多头自注意力
        # multi_out = self.fuse_strategy_attention(text_out, img_out)

        # 融合策略3：使用封装的transformer_encoder
        multi_out = self.fuse_strategy_transformer(text_out, img_out)

        # 分类器
        text_out = self.classifier_text(text_out)
        img_out = self.classifier_img(img_out)
        multi_out = self.classifier_multi(multi_out)
        return text_out, img_out, multi_out

    def fuse_strategy_transformer(self, text_out, img_out):
        multimodal_sequence = torch.stack((text_out, img_out), dim=1)  # N, L, E
        return self.transformer_encoder(multimodal_sequence)

    def fuse_strategy_attention(self, text_out, img_out):
        # 引入线性变换
        text_k1 = self.linear_text_k1(text_out)
        text_v1 = self.linear_text_v1(text_out)
        img_k2 = self.linear_img_k2(img_out)
        img_v2 = self.linear_img_v2(img_out)
        # 生成key, value, query
        key = torch.stack((text_k1, img_k2), dim=1)  # N, L, E
        value = torch.stack((text_v1, img_v2), dim=1)  # N, L, E
        query = torch.stack((text_out, img_out), dim=1)  # N, L, E
        # 多头自注意力
        attn_output, attn_output_weights = self.multihead_attn(query, key, value)
        return attn_output
