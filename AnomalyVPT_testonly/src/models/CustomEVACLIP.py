from torch import nn
import torch
import torch.nn.functional as F

from src.eva_clip import get_tokenizer
from src.models.SegDecoder import SegDecoder


class PromptOrder:
    def __init__(self) -> None:
        super().__init__()
        self.state_normal_list = [
            "{}",
            "flawless {}",
            "perfect {}",
            "unblemished {}",
            "{} without flaw",
            "{} without defect",
            "{} without damage"
        ]

        self.state_anomaly_list = [
            "damaged {}",
            "{} with flaw",
            "{} with defect",
            "{} with damage"
        ]

        self.template_list = [
            "a cropped photo of the {}.",
            "a close-up photo of a {}.",
            "a close-up photo of the {}.",
            "a bright photo of a {}.",
            "a bright photo of the {}.",
            "a dark photo of the {}.",
            "a dark photo of a {}.",
            "a jpeg corrupted photo of the {}.",
            "a jpeg corrupted photo of the {}.",
            "a blurry photo of the {}.",
            "a blurry photo of a {}.",
            "a photo of a {}.",
            "a photo of the {}.",
            "a photo of a small {}.",
            "a photo of the small {}.",
            "a photo of a large {}.",
            "a photo of the large {}.",
            "a photo of the {} for visual inspection.",
            "a photo of a {} for visual inspection.",
            "a photo of the {} for anomaly detection.",
            "a photo of a {} for anomaly detection."
        ]

    def prompt(self, class_name='object'):
        # object, product, item, target, {none}
        class_state = [ele.format(class_name) for ele in self.state_normal_list]
        normal_ensemble_template = [class_template.format(ele) for ele in class_state for class_template in
                                    self.template_list]
        class_state = [ele.format(class_name)
                       for ele in self.state_anomaly_list]
        anomaly_ensemble_template = [class_template.format(ele) for ele in class_state for class_template in
                                     self.template_list]
        return [normal_ensemble_template, anomaly_ensemble_template]


def harmonic_mean_images(images, eps=1e-7):
    print(f"images shape: {images.shape}")
    reciprocal_channels = 1 / (images + eps)
    mean_reciprocal = torch.mean(
        reciprocal_channels, dim=1, keepdim=True)
    harmonic_mean = torch.reciprocal(mean_reciprocal)
    return harmonic_mean.squeeze()  # [bs, h, w]

def harmonic_mean_image_list(image_list, eps=1e-7):
    '''
        image_list: [n, B, image_H, image_W, 2]
    '''
    harmonic_mean_denominator = torch.sum(1 / (image_list + eps), dim=0)
    n = image_list.shape[0]
    return n / harmonic_mean_denominator # [n, B, image_H, image_W, 2]


def aggregate_fpn_logits(image_list, eps=1e-7):
    '''
        image_list: list of [B, image_H, image_W, 2]
    '''
    normalized_seg_maps = [F.normalize(seg_map, p=1, dim=-1) for seg_map in image_list]
    normalized_seg_maps_stacked = torch.stack(normalized_seg_maps) # [lens, B, image_H, image_W, 2]

    return torch.max(normalized_seg_maps_stacked, dim=0).values

def calc_anomaly_map(image_features, text_features, patch_size=14, img_size=336, scale=1 / 0.07):
    num_patches = int((img_size // patch_size) ** 2)
    text_features = text_features.expand(image_features.shape[0], 2, image_features.shape[2])  # [B, 2, C]
    feats = torch.bmm(image_features, text_features.transpose(1, 2))
    sim = feats * scale
    sim = sim[:, :num_patches, :]  # sim = [batch_size, num_patches, 2]
    side = int(img_size // patch_size)  # side
    sim = sim.reshape(sim.shape[0], side, side, -1).permute(0, 3, 1, 2)
    sim = torch.nn.functional.interpolate(sim, img_size, mode='bilinear')
    sim = sim.permute(0, 2, 3, 1)
    return sim # [B, image_H, image_W, 2]


class OrthogonalRegularization(nn.Module):
    def __init__(self, weight, decay=0.001):
        super(OrthogonalRegularization, self).__init__()
        self.weight = weight
        self.decay = decay

    def forward(self):
        if isinstance(self.weight, nn.Parameter):
            inner_products = torch.mm(self.weight, self.weight.t())
            identity_matrix = torch.eye(inner_products.size(0), device=inner_products.device)
            ortho_loss = torch.norm(inner_products - identity_matrix, p='fro')
        elif isinstance(self.weight, nn.ParameterList):
            ortho_loss = 0
            for param in self.weight:
                inner_products = torch.mm(param, param.t())
                identity_matrix = torch.eye(inner_products.size(0), device=inner_products.device)
                ortho_loss += torch.norm(inner_products - identity_matrix, p='fro')
        else:
            raise AttributeError("Weight is not nn.Parameter")
        return self.decay * ortho_loss


class CustomEVACLIP(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        self.clip_model = clip_model
        embed_size = self.clip_model.text.output_dim
        self.dtype = self.clip_model.logit_scale.dtype
        text_prompts = PromptOrder().prompt()
        self.text_features = nn.Parameter(self.build_ensemble_text_features(text_prompts, dtype=self.dtype))
        self.logit_scale = self.clip_model.logit_scale
        # self.ortho_reg = OrthogonalRegularization(self.clip_model.visual.learnable_prompt.p)

        # pixel level
        self.image_size = cfg.INPUT.SIZE
        self.patch_size = cfg.MODEL.PATCH_SIZE
        num_patches = (self.image_size[0] // self.patch_size) * (self.image_size[1] // self.patch_size)

        prompt_length = cfg.MODEL.VP_LENGTH
        self.seg_decoder = SegDecoder(input_dim=num_patches + prompt_length,
                                      output_dim=num_patches + prompt_length,
                                      embed_size=embed_size,
                                      size=4)

        from src.models.SegDecoder import QuickGELU
        
        self.fpn_scale = [2, 1, 1/2, 1/4]
        self.fpn_decoder = nn.ModuleList([nn.Sequential(
                nn.Linear(embed_size, 8),
                QuickGELU(),
                nn.Linear(8, embed_size),
            ) for _ in self.fpn_scale])
        # TODO: check device!!
        # self.simple_text_encoder = SimpleTextEncoder(clip_model)

    @torch.no_grad()
    def build_ensemble_text_features(self, text_prompts, dtype, model_name="EVA02-CLIP-L-14-336"):
        tokenizer = get_tokenizer(model_name)
        text_features = torch.empty(0)
        for templates in text_prompts:
            tokens = tokenizer(templates)
            ensemble_text_features = self.clip_model.encode_text(tokens)
            # for prompt in templates:
            #     tokens = tokenizer.tokenize(prompt)
            #     cur_embed = self.clip_model.encode_text(tokens).type(dtype=dtype)
            #     ensemble_text_features = torch.cat([ensemble_text_features, cur_embed], dim=0)
            
            avg_text_features = torch.mean(ensemble_text_features, dim=0, keepdim=True)
            text_features = torch.cat([text_features, avg_text_features])

        return text_features

    def forward(self, image, is_train=False, up=True, impaths=None):
        output = dict()
        # b, c, h, w = image.shape
        text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
        res = self.clip_model.encode_image(image.type(self.dtype))
        origin_image_features = res['pooled']

        image_features = origin_image_features / origin_image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        raw_score = image_features @ text_features.t()
        logits = logit_scale * raw_score

        output['logits'] = logits
        output['mid_logits'] = [logits]

        output['logits'] = logits

        expand_text_features = text_features.repeat(image_features.shape[0], 1, 1)
        if up:
                       
            patch_features = res['tokens']
            patch_features = self.seg_decoder(patch_features, expand_text_features, idx=-1)  # cross with text
            patch_features = patch_features / patch_features.norm(dim=-1, keepdim=True)
            anomaly_map = calc_anomaly_map(patch_features, text_features, img_size=self.image_size[0],
                                           scale=logit_scale)

            output['map'] = anomaly_map
            output['mid_map'] = []
            # output['out_map'] = harmonic_mean_images(stack_images(anomaly_map, []))
            output['out_map'] = anomaly_map

        return output
