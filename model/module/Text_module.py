from torch import nn
from transformers import RobertaModel


class TextModule(nn.Module):
    def __init__(self, args):
        super(TextModule, self).__init__()
        self.encoder = RobertaModel.from_pretrained(args.pretrained_model)
        # True:每个参数都要求梯度；False:冻结参数
        for param in self.encoder.parameters():
            param.requires_grad = True
        self.transform = nn.Sequential(
            nn.Linear(768, 1000),
            nn.ReLU(),
        )

    def forward(self, encoded_input):
        encoder_output = self.encoder(**encoded_input)
        hidden_state = encoder_output['last_hidden_state']
        pooler_output = encoder_output['pooler_output']
        output = self.transform(pooler_output)
        # hidden_state ([N, L, E])
        # N:bach大小, L:序列长度, E:特征维度
        return hidden_state, output
