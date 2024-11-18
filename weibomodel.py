from __future__ import print_function
import math
import random
# from random import random
import torch
import torch.nn as nn
from torch.distributions import Normal, Independent
from torch.nn.functional import softplus
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal
import numpy as np




class Expert(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Expert, self).__init__()

        p = 0
        expert_hidden_layers = [64, 32]
        self.expert_layer = nn.Sequential(
            nn.Linear(input_dim, expert_hidden_layers[0]),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(expert_hidden_layers[0], expert_hidden_layers[1]),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(expert_hidden_layers[1], output_dim),
            nn.ReLU(),
            nn.Dropout(p)
        )

    def forward(self, x):
        out = self.expert_layer(x)
        return out


class Expert_Gate(nn.Module):
    def __init__(self, feature_dim, expert_dim, n_expert, n_task,
                 use_gate=True):
        super(Expert_Gate, self).__init__()
        self.n_task = n_task
        self.use_gate = use_gate

        '''专家网络'''
        for i in range(n_expert):
            setattr(self, "expert_layer" + str(i + 1), Expert(feature_dim, expert_dim))
        self.expert_layers = [getattr(self, "expert_layer" + str(i + 1)) for i in range(n_expert)]

        '''门控网络'''
        for i in range(n_task):
            setattr(self, "gate_layer" + str(i + 1), nn.Sequential(nn.Linear(feature_dim, n_expert),
                                                                   nn.Softmax(dim=1)))
        self.gate_layers = [getattr(self, "gate_layer" + str(i + 1)) for i in range(n_task)]

    def forward(self, x):
        if self.use_gate:
            # 构建多个专家网络
            E_net = [expert(x) for expert in self.expert_layers]
            E_net = torch.cat(([e[:, np.newaxis, :] for e in E_net]), dim=1)

            # 构建多个门网络
            gate_net = [gate(x) for gate in self.gate_layers]

            # towers计算：对应的门网络乘上所有的专家网络
            towers = []
            for i in range(self.n_task):
                g = gate_net[i].unsqueeze(2)
                tower = torch.matmul(E_net.transpose(1, 2), g)
                towers.append(tower.transpose(1, 2).squeeze(1))
        else:
            E_net = [expert(x) for expert in self.expert_layers]
            towers = sum(E_net) / len(E_net)
        return towers


class MMoE(nn.Module):

    def __init__(self, feature_dim=128, expert_dim=64, n_expert=4, n_task=2, use_gate=True):
        super(MMoE, self).__init__()

        self.use_gate = use_gate
        self.Expert_Gate = Expert_Gate(feature_dim=feature_dim, expert_dim=expert_dim, n_expert=n_expert, n_task=n_task,
                                       use_gate=use_gate)

        '''Tower1'''
        p1 = 0
        hidden_layer1 = [64, 32]  # [64,32]
        self.tower1 = nn.Sequential(
            nn.Linear(expert_dim, hidden_layer1[0]),
            nn.ReLU(),
            nn.Dropout(p1),
            nn.Linear(hidden_layer1[0], hidden_layer1[1]),
            nn.ReLU(),
            nn.Dropout(p1),
            nn.Linear(hidden_layer1[1], 128))
        '''Tower2'''
        p2 = 0
        hidden_layer2 = [64, 32]
        self.tower2 = nn.Sequential(
            nn.Linear(expert_dim, hidden_layer2[0]),
            nn.ReLU(),
            nn.Dropout(p2),
            nn.Linear(hidden_layer2[0], hidden_layer2[1]),
            nn.ReLU(),
            nn.Dropout(p2),
            nn.Linear(hidden_layer2[1], 128))

    def forward(self, x):

        towers = self.Expert_Gate(x)
        if self.use_gate:
            out1 = self.tower1(towers[0])
            out2 = self.tower2(towers[1])
        else:
            out1 = self.tower1(towers)
            out2 = self.tower2(towers)

        return out1, out2


class FastCNN(nn.Module):

    def __init__(self, channel=32,
                 kernel_size=(1, 2, 4, 8)):
        super(FastCNN, self).__init__()
        self.fast_cnn = nn.ModuleList()
        for kernel in kernel_size:
            self.fast_cnn.append(
                nn.Sequential(
                    nn.Conv1d(200, channel, kernel_size=kernel),
                    nn.BatchNorm1d(channel),
                    nn.ReLU(),
                    nn.AdaptiveMaxPool1d(1)
                )
            )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x_out = []
        for module in self.fast_cnn:
            x_out.append(module(x).squeeze())
        x_out = torch.cat(x_out, -1)
        return x_out


class EncodingPart(nn.Module):
    def __init__(
            self,
            cnn_channel=32,
            cnn_kernel_size=(1, 2, 4, 8),
            shared_image_dim=128,
            shared_text_dim=128
    ):
        super(EncodingPart, self).__init__()
        self.shared_text_encoding = FastCNN(
            channel=cnn_channel,
            kernel_size=cnn_kernel_size
        )
        self.shared_text_linear = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, shared_text_dim),
            nn.BatchNorm1d(shared_text_dim),
            nn.ReLU()
        )
        self.shared_image = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, shared_image_dim),
            nn.BatchNorm1d(shared_image_dim),
            nn.ReLU()
        )

    def forward(self, text, image):
        text_encoding = self.shared_text_encoding(text)
        text_shared = self.shared_text_linear(text_encoding)
        image_shared = self.shared_image(image)
        return text_shared, image_shared


class SimilarityModule(nn.Module):  # 模态对齐部分的相似度模型
    def __init__(self, shared_dim=128, sim_dim=64):
        super(SimilarityModule, self).__init__()
        self.encoding = EncodingPart()
        self.text_MMoER = MMoE()
        self.image_MMoER = MMoE()
        self.text_aligner = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, sim_dim),
            nn.BatchNorm1d(sim_dim),
            nn.ReLU()
        )
        self.image_aligner = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, sim_dim),
            nn.BatchNorm1d(sim_dim),
            nn.ReLU()
        )
        self.sim_classifier_dim = sim_dim * 2
        self.sim_classifier = nn.Sequential(
            nn.BatchNorm1d(self.sim_classifier_dim),
            nn.Linear(self.sim_classifier_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, text, image):
        text_encoding, image_encoding = self.encoding(text, image)
        text_MMoE1, text_MMoE2 = self.text_MMoER(text_encoding)
        image_MMoE1, image_MMoE2 = self.text_MMoER(image_encoding)


        text_aligned = self.text_aligner(text_MMoE1)
        image_aligned = self.image_aligner(image_MMoE1)

        sim_feature = torch.cat([text_aligned, image_aligned], 1)
        pred_similarity = self.sim_classifier(sim_feature)
        return text_aligned, image_aligned, pred_similarity


class Encoder(nn.Module):
    def __init__(self, z_dim=2):
        super(Encoder, self).__init__()
        self.z_dim = z_dim
        # Vanilla MLP
        self.net = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(True),
            nn.Linear(64, z_dim * 2),
        )

    def forward(self, x):
        params = self.net(x)
        mu, sigma = params[:, :self.z_dim], params[:, self.z_dim:]
        sigma = softplus(sigma) + 1e-7
        return Independent(Normal(loc=mu, scale=sigma), 1)


class AmbiguityLearning(nn.Module):
    def __init__(self):
        super(AmbiguityLearning, self).__init__()
        self.encoding = EncodingPart()
        self.encoder_text = Encoder()
        self.encoder_image = Encoder()

    def forward(self, text_MMoE1, image_MMoE1):
        p_z1_given_text = self.encoder_text(text_MMoE1)
        p_z2_given_image = self.encoder_image(image_MMoE1)
        z1 = p_z1_given_text.rsample()
        z2 = p_z2_given_image.rsample()
        kl_1_2 = p_z1_given_text.log_prob(z1) - p_z2_given_image.log_prob(z1)
        kl_2_1 = p_z2_given_image.log_prob(z2) - p_z1_given_text.log_prob(z2)
        skl = (kl_1_2 + kl_2_1) / 2.
        skl = nn.functional.sigmoid(skl)
        return skl


class UnimodalDetection(nn.Module):
    def __init__(self, shared_dim=128, prime_dim=16):
        super(UnimodalDetection, self).__init__()
        self.text_uni = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, prime_dim),
            nn.BatchNorm1d(prime_dim),
            nn.ReLU()
        )
        self.image_uni = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, prime_dim),
            nn.BatchNorm1d(prime_dim),
            nn.ReLU()
        )

    def forward(self, text_MMoE1, image_MMoE1):
        text_prime = self.text_uni(text_MMoE1)
        image_prime = self.image_uni(image_MMoE1)
        return text_prime, image_prime

class SubNet(nn.Module):
    '''
    The subnetwork that is used in LMF for video and image in the pre-fusion stage
    '''

    def __init__(self, in_size, hidden_size, dropout):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            dropout: dropout probability
        Output:
            (return value in forward) a tensor of shape (batch_size, hidden_size)
        '''
        super(SubNet, self).__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, in_size)
        '''
        normed = self.norm(x)
        dropped = self.drop(normed)
        y_1 = F.relu(self.linear_1(dropped))
        y_2 = F.relu(self.linear_2(y_1))
        y_3 = F.relu(self.linear_3(y_2))

        return y_3


class TextSubNet(nn.Module):
    '''
    The LSTM-based subnetwork that is used in LMF for text
    '''

    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super(TextSubNet, self).__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, sequence_len, in_size)
        '''
        final_states = self.rnn(x)
        h = self.dropout(final_states[0].squeeze())
        y_1 = self.linear_1(h)
        return y_1


class LMF(nn.Module):
    '''
    Low-rank Multimodal Fusion
    '''

    def __init__(self, text_out=64, hidden_dims=[16, 64], dropouts=[0, 0.1, 0.15, 0.2, 0.3, 0.5], output_dim=64, rank=4, text_in_dim=64, image_in_dim=64, use_softmax=False):
        # 原来是0.2，0.1，0.5
        '''
        Args:
            input_dims - a length-3 tuple, contains (image_dim, video_dim, text_dim)
            hidden_dims - another length-3 tuple, hidden dims of the sub-networks
            text_out - int, specifying the resulting dimensions of the text subnetwork
            dropouts - a length-4 tuple, contains (image_dropout, video_dropout, text_dropout, post_fusion_dropout)
            output_dim - int, specifying the size of output
            rank - int, specifying the size of rank in LMF
        Output:
            (return value in forward) a scalar value between -3 and 3
        '''
        super(LMF, self).__init__()

        # dimensions are specified in the order of image, video and text
        self.image_in = image_in_dim
        self.text_in = text_in_dim

        self.image_hidden = hidden_dims[0]
        self.text_hidden = hidden_dims[1]

        self.text_out = text_out
        self.output_dim = output_dim

        self.rank = rank
        self.use_softmax = use_softmax

        self.image_prob = dropouts[0]

        self.text_prob = dropouts[1]
        self.post_fusion_prob = dropouts[2]

        # define the pre-fusion subnetworks
        self.image_subnet = SubNet(self.image_in, self.image_hidden, self.image_prob)
        self.text_subnet = TextSubNet(self.text_in, self.text_hidden, self.text_out, dropout=self.text_prob)
       # self.text_subnet = SubNet(self.text_in, self.text_hidden, self.text_prob) #没有用文章里面原来构建的文本处理框架

        # define the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
        # self.post_fusion_layer_1 = nn.Linear((self.text_out + 1) * (self.video_hidden + 1) * (self.image_hidden + 1), self.post_fusion_dim)
        self.image_factor = Parameter(torch.Tensor(self.rank, self.image_hidden + 1, self.output_dim))

        self.text_factor = Parameter(torch.Tensor(self.rank, self.text_out + 1, self.output_dim))
        self.fusion_weights = Parameter(torch.Tensor(1, self.rank))
        self.fusion_bias = Parameter(torch.Tensor(1, self.output_dim))

        # init teh factors
        nn.init.xavier_normal_(self.image_factor)

        nn.init.xavier_normal_(self.text_factor)
        nn.init.xavier_normal_(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

    def forward(self, image_MMoE2, text_MMoE2):
        '''
        Args:
            image_x: tensor of shape (batch_size, image_in)
            text_x: tensor of shape (batch_size, sequence_len, text_in)
        '''
        image_h = self.image_subnet(image_MMoE2)

        text_h = self.text_subnet(text_MMoE2)
        batch_size = image_h.data.shape[0]

        # next we perform low-rank multimodal fusion
        # here is a more efficient implementation than the one the paper describes
        # basically swapping the order of summation and elementwise product
        if image_h.is_cuda:
            DTYPE = torch.cuda.FloatTensor
        else:
            DTYPE = torch.FloatTensor

        _image_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), image_h), dim=1)

        _text_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), text_h), dim=1)

        fusion_image = torch.matmul(_image_h, self.image_factor)

        fusion_text = torch.matmul(_text_h, self.text_factor)
        fusion_zy = fusion_image * fusion_text

        # output = torch.sum(fusion_zy, dim=0).squeeze()
        # use linear transformation instead of simple summation, more flexibility
        output = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias
        output = output.view(-1, self.output_dim)
        if self.use_softmax:
            output = F.softmax(output)
        return output


class DetectionModule(nn.Module):
    def __init__(self, feature_dim=64 + 16 + 16, h_dim=64):
        super(DetectionModule, self).__init__()
        self.encoding = EncodingPart()
        self.ambiguity_module = AmbiguityLearning()
        self.uni_repre = UnimodalDetection()
        self.cross_module = LMF()
        self.classifier_corre = nn.Sequential(
            nn.Linear(feature_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(h_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(h_dim, 2)
        )

    def forward(self, text_raw, image_raw, text_MMoE2, image_MMoE2):
        # text_encoding, image_encoding = self.encoding_module(text, image)
        skl = self.ambiguity_module(text_MMoE2, image_MMoE2)
        text_prime, image_prime = self.encoding(text_raw, image_raw)
        text_prime, image_prime = self.uni_repre(text_prime, image_prime)
        correlation = self.cross_module(text_MMoE2, image_MMoE2)
        weight_uni = (1 - skl).unsqueeze(1)
        weight_corre = skl.unsqueeze(1)
        text_final = weight_uni * text_prime
        img_final = weight_uni * image_prime
        corre_final = weight_corre * correlation
        final_corre = torch.cat([text_final, img_final, corre_final], 1)
        pre_label = self.classifier_corre(final_corre)
        return pre_label

