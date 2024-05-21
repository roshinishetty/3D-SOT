from torch import nn

## P2B backbone, xcorr and rpn
from models.backbone.pointnet import Pointnet_Backbone
from models.head.xcorr import P2B_XCorr
from models.head.rpn import P2BVoteNetRPN

## Our backbone
from models.backbone.pointnet2 import Pointnet_Plus_Plus


from models import base_model
from collections import defaultdict
import torch
import torch.nn.functional as F

## OSP2B transformer imports, we are not using this
from .transformer import TransformerDecoder, TransformerEncoder
from .multihead_attention import MultiheadAttention

## Our transformer
from .customtransformer import TransformerFusion

from pointnet2.utils import pytorch_utils as pt_utils
import pickle


## Pos Embed code from Osp2b
class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, input_channel=3, num_pos_feats=256):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

        for m in self.position_embedding_head.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.uniform_()
                m.bias.data.zero_()

    def forward(self, xyz):
        # xyz : BxNx3
        xyz = xyz.transpose(1, 2).contiguous()
        # Bx3xN
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding

#P2B architecture modified by adding transformers(replacing xcorr)
class P2B(base_model.MatchingBaseModel):
    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        self.save_hyperparameters()
        
        ## P2B backbone init
        self.backbone = Pointnet_Backbone(self.config.use_fps, self.config.normalize_xyz, return_intermediate=False)
        ## Our backbone init
        # self.backbone = Pointnet_Plus_Plus(self.config.use_fps, self.config.normalize_xyz, return_intermediate=False)
        
        self.conv_final = nn.Conv1d(256, self.config.feature_channel, kernel_size=1)

        ## P2B xcorr, we are not using this, but not commenting it as our model weights files have weights of this network (during training) and commenting this causes error
        if config.category_name=="Car":
            self.xcorr = P2B_XCorr(feature_channel=self.config.feature_channel,
                                   hidden_channel=self.config.hidden_channel,
                                   out_channel=self.config.out_channel)
        self.rpn = P2BVoteNetRPN(self.config.feature_channel,
                                 vote_channel=self.config.vote_channel,
                                 num_proposal=self.config.num_proposal,
                                 normalize_xyz=self.config.normalize_xyz)
        
        ## our transformer network initializtion
        self.transformerfusion = TransformerFusion(num_layers_encoder = 1, num_layers_decoder=1, d_model=256, d_internal=128, n_heads=1)
        
        ##############################################################################
        ## OSP2B transformer config, experimented with this to compare our results
        
#         d_model = 256
#         num_layers = 1
#         self.with_pos_embed = True
#         self.fea_layer = (pt_utils.Seq(256)
#                 .conv1d(256, bn=True)
#                 .conv1d(256, activation=None))
        
#         multihead_attn = MultiheadAttention(
#             feature_dim=d_model, n_head=1, key_feature_dim=128)
        
#         if self.with_pos_embed:
#             encoder_pos_embed = PositionEmbeddingLearned(3, d_model)
#             decoder_pos_embed = PositionEmbeddingLearned(3, d_model)
#         else:
#             encoder_pos_embed = None
#             decoder_pos_embed = None

#         self.encoder = TransformerEncoder(
#             multihead_attn=multihead_attn, FFN=None,
#             d_model=d_model, num_encoder_layers=num_layers,
#             self_posembed=encoder_pos_embed)
#         self.decoder = TransformerDecoder(
#             multihead_attn=multihead_attn, FFN=None,
#             d_model=d_model, num_decoder_layers=num_layers,
#             self_posembed=decoder_pos_embed)
        
    ## OSP2B transformer fusion
    def transform_fuse(self, search_feature, search_xyz,
                       template_feature, template_xyz):
        """Use transformer to fuse feature.

        template_feature : BxCxN
        template_xyz : BxNx3
        """
        # print(template_xyz.shape,template_feature.shape,search_xyz.shape,search_feature.shape)
        # torch.Size([1, 64, 3]) torch.Size([1, 256, 64]) torch.Size([1, 128, 3]) torch.Size([1, 256, 128])

        # BxCxN -> NxBxC
        search_feature = search_feature.permute(2, 0, 1)
        template_feature = template_feature.permute(2, 0, 1)

        num_img_train = search_feature.shape[0]
        num_img_template = template_feature.shape[0]

        ## encoder
        encoded_memory = self.encoder(template_feature,
            query_pos=template_xyz if self.with_pos_embed else None) #(64,1,256)

        encoded_feat = self.decoder(search_feature,
                                    memory=encoded_memory,
                                    query_pos=search_xyz) #(128,1,256)

        # NxBxC -> BxCxN
        encoded_feat = encoded_feat.permute(1, 2, 0)
        encoded_feat = self.fea_layer(encoded_feat)

        return encoded_feat        
###################################################################################

    def forward(self, input_dict):
        """
        :param input_dict:
        {
        'template_points': template_points.astype('float32'),
        'search_points': search_points.astype('float32'),
        'box_label': np.array(search_bbox_reg).astype('float32'),
        'bbox_size': search_box.wlh,
        'seg_label': seg_label.astype('float32'),
        }

        :return:
        """
        template = input_dict['template_points']
        search = input_dict['search_points']
        M = template.shape[1]
        N = search.shape[1]
        template_xyz, template_feature, _ = self.backbone(template, [M // 2, M // 4, M // 8])
        search_xyz, search_feature, sample_idxs = self.backbone(search, [N // 2, N // 4, N // 8])
        template_feature = self.conv_final(template_feature)
        search_feature = self.conv_final(search_feature)
        
        ## OSP2B transformer
        # fusion_feature = self.transform_fuse(
        #     search_feature, search_xyz, template_feature, template_xyz)
        
        ## Our transformer
        fusion_feature, probs = self.transformerfusion(search_feature, search_xyz, template_feature, template_xyz)
        
        ## Saving weights for attention visuaization
        # dbfile = open(f"probs", 'wb')
        # probs_dict = {"probs":probs.cpu().numpy(), "search_xyz":search_xyz.cpu().numpy(), "template_xyz":template_xyz.cpu().numpy()}
        # pickle.dump(probs_dict, dbfile)
        
        
        ## P2B xcorr
        # fusion_feature = self.xcorr(template_feature, search_feature, template_xyz)
        
        estimation_boxes, estimation_cla, vote_xyz, center_xyzs = self.rpn(search_xyz, fusion_feature)
        end_points = {"estimation_boxes": estimation_boxes,
                      "vote_center": vote_xyz,
                      "pred_seg_score": estimation_cla,
                      "center_xyz": center_xyzs,
                      'sample_idxs': sample_idxs,
                      'estimation_cla': estimation_cla,
                      "vote_xyz": vote_xyz
                      }
        return end_points

    def training_step(self, batch, batch_idx):
        """
        {"estimation_boxes": estimation_boxs.transpose(1, 2).contiguous(),
                  "vote_center": vote_xyz,
                  "pred_seg_score": estimation_cla,
                  "center_xyz": center_xyzs,
                  "seed_idxs":
                  "seg_label"
        }
        """
        end_points = self(batch)
        estimation_cla = end_points['estimation_cla']  # B,N
        N = estimation_cla.shape[1]
        seg_label = batch['seg_label']
        sample_idxs = end_points['sample_idxs']  # B,N
        # update label
        seg_label = seg_label.gather(dim=1, index=sample_idxs[:, :N].long())  # B,N
        batch["seg_label"] = seg_label
        # compute loss
        loss_dict = self.compute_loss(batch, end_points)
        loss = loss_dict['loss_objective'] * self.config.objectiveness_weight \
               + loss_dict['loss_box'] * self.config.box_weight \
               + loss_dict['loss_seg'] * self.config.seg_weight \
               + loss_dict['loss_vote'] * self.config.vote_weight
        self.log('loss/train', loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=False)
        self.log('loss_box/train', loss_dict['loss_box'].item(), on_step=True, on_epoch=True, prog_bar=True,
                 logger=False)
        self.log('loss_seg/train', loss_dict['loss_seg'].item(), on_step=True, on_epoch=True, prog_bar=True,
                 logger=False)
        self.log('loss_vote/train', loss_dict['loss_vote'].item(), on_step=True, on_epoch=True, prog_bar=True,
                 logger=False)
        self.log('loss_objective/train', loss_dict['loss_objective'].item(), on_step=True, on_epoch=True, prog_bar=True,
                 logger=False)
        self.logger.experiment.add_scalars('loss', {'loss_total': loss.item(),
                                                    'loss_box': loss_dict['loss_box'].item(),
                                                    'loss_seg': loss_dict['loss_seg'].item(),
                                                    'loss_vote': loss_dict['loss_vote'].item(),
                                                    'loss_objective': loss_dict['loss_objective'].item()},
                                           global_step=self.global_step)

        return loss
