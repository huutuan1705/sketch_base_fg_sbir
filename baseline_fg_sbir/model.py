import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch import optim
from tqdm import tqdm

from backbones import InceptionV3
from attention import Linear_global, SelfAttention, Attention_global

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FGSBIR_Model(nn.Module):
    def __init__(self, args):
        super(FGSBIR_Model, self).__init__()
        self.sample_embedding_network = eval(args.backbone_name + "(args)")
        self.loss = nn.TripletMarginLoss(margin=args.margin)        
        self.sample_train_params = self.sample_embedding_network.parameters()
        self.args = args
        
        def init_weights(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                nn.init.kaiming_normal_(m.weight)
        
        self.attention = SelfAttention(args)
        self.attn_params = self.attention.parameters()
        
        self.linear = Linear_global(feature_num=self.args.output_size)
        self.linear_params = self.linear.parameters()
        
        if self.args.use_kaiming_init:
            self.attention.apply(init_weights)
            self.linear.apply(init_weights)
            
        self.optimizer = optim.Adam([
            {'params': self.sample_embedding_network.parameters(), 'lr': args.learning_rate},
        ])
        
    def train_model(self, batch):
        self.train()
        self.optimizer.zero_grad()
            
        positive_features = self.linear(self.attention(
            self.sample_embedding_network(batch['positive_img'].to(device))
        )) # (N, 64)
        negative_features = self.linear(self.attention(
            self.sample_embedding_network(batch['negative_img'].to(device))
        )) # (N, 64)

        loss = 0
        # print("len(batch['sketch_imgs']): ", len(batch['sketch_imgs'])) # 64
        for i in range(len(batch['sketch_imgs'])):
            sketch_features = self.linear(self.attention(
                self.sample_embedding_network(batch['sketch_imgs'][i].to(device)))) # (25,  64)
            
            positive_feature = positive_features[i]
            negative_feature = negative_features[i]
            
            loss += self.loss(sketch_features, positive_feature.unsqueeze(0), negative_feature.unsqueeze(0))
            
        loss.backward()
        self.optimizer.step()

        return loss.item() 

    
    def evaluate(self, dataloader_test):
        self.eval()
        sketch_array_tests = []
        sketch_names = []
        image_array_tests = torch.FloatTensor().to(device)
        image_names = []
        
        for i_batch, batch in enumerate(tqdm(dataloader_test)):
            sketch_features_all = torch.FloatTensor().to(device)
            for data_sketch in batch['sketch_imgs']:
                # print(data_sketch.shape) # (1, 3, 299, 299)
                sketch_feature = self.attention(
                    self.sample_embedding_network(data_sketch.to(device))
                )
                # print("sketch_feature.shape: ", sketch_feature.shape) #(1, 2048)
                sketch_features_all = torch.cat((sketch_features_all, sketch_feature.detach()))
            
            # print("sketch_feature_ALL.shape: ", sketch_features_all.shape) # (25, 2048)           
            sketch_array_tests.append(sketch_features_all.cpu())
            sketch_names.extend(batch['sketch_path'])
            
            if batch['positive_path'][0] not in image_names:
                positive_feature = self.linear(self.attention(
                    self.sample_embedding_network(batch['positive_img'].to(device))
                ))
                image_array_tests = torch.cat((image_array_tests, positive_feature))
                image_names.extend(batch['positive_path'])
        
        num_steps = len(sketch_array_tests[0])
        avererage_area = []
        avererage_area_percentile = []
                
        rank_all = torch.zeros(len(sketch_array_tests), num_steps)
        rank_all_percentile = torch.zeros(len(sketch_array_tests), num_steps)
                
        for i_batch, sampled_batch in enumerate(sketch_array_tests):
            mean_rank = []
            mean_rank_percentile = []
            sketch_name = sketch_names[i_batch]
            
            sketch_query_name = '_'.join(sketch_name.split('/')[-1].split('_')[:-1])
            position_query = image_names.index(sketch_query_name)
            
            for i_sketch in range(sampled_batch.shape[0]):
                target_distance = F.pairwise_distance(sampled_batch.unsqueeze(0).to(device), image_array_tests[position_query].unsqueeze(0).to(device))
                distance = F.pairwise_distance(sampled_batch.unsqueeze(0).to(device), image_array_tests.to(device))
                
                rank_all[i_batch, i_sketch] = distance.le(target_distance).sum()

                rank_all_percentile[i_batch, i_sketch] = (len(distance) - rank_all[i_batch, i_sketch]) / (len(distance) - 1)
                if rank_all[i_batch, i_sketch].item() == 0:
                    mean_rank.append(1.)
                else:
                    mean_rank.append(1/rank_all[i_batch, i_sketch].item())
                        #1/(rank)
                    mean_rank_percentile.append(rank_all_percentile[i_batch, i_sketch].item())
            
            avererage_area.append(np.sum(mean_rank)/len(mean_rank))
            avererage_area_percentile.append(np.sum(mean_rank_percentile)/len(mean_rank_percentile))
            
        top1_accuracy = rank_all[:, -1].le(1).sum().numpy() / rank_all.shape[0]
        top5_accuracy = rank_all[:, -1].le(5).sum().numpy() / rank_all.shape[0]
        top10_accuracy = rank_all[:, -1].le(10).sum().numpy() / rank_all.shape[0]
        
        meanMA = np.mean(avererage_area_percentile)
        meanMB = np.mean(avererage_area)
        
        return top1_accuracy, top5_accuracy, top10_accuracy, meanMA, meanMB