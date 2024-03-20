import torch

def cross_destillation_loss():
    pass


#def gradcam_distillation(gradients_a, gradients_b, activations_a, activations_b, factor=1):
#    """Distillation loss between gradcam-generated attentions of two models.
#    References:
#        * Dhar et al.
#          Learning without Memorizing
#          CVPR 2019
#    :param base_logits: [description]
#    :param list_attentions_a: [description]
#    :param list_attentions_b: [description]
#    :param factor: [description], defaults to 1
#    :return: [description
#    https://github.com/arthurdouillard/incremental_learning.pytorch/blob/master/inclearn/lib/losses/distillation.py
#    """
#    attentions_a = _compute_gradcam_attention(gradients_a, activations_a)
#    attentions_b = _compute_gradcam_attention(gradients_b, activations_b)#

#    assert len(attentions_a.shape) == len(attentions_b.shape) == 4#
#    assert attentions_a.shape == attentions_b.shap

#    batch_size = attentions_a.shape[0]

#    flat_attention_a = F.normalize(attentions_a.view(batch_size, -1), p=2, dim=-1)
#    flat_attention_b = F.normalize(attentions_b.view(batch_size, -1), p=2, dim=-1)#

#    distances = torch.abs(flat_attention_a - flat_attention_b).sum(-1)

#    return factor * torch.mean(distances)

def mer_loss(new_logits, old_logits):
    """Distillation loss that is less important if the new model is unconfident.
    Reference:
        * Kim et al.
          Incremental Learning with Maximum Entropy Regularization: Rethinking
          Forgetting and Intransigence.
    :param new_logits: Logits from the new (student) model.
    :param old_logits: Logits from the old (teacher) model.
    :return: A float scalar loss.
    """
    new_probs = F.softmax(new_logits, dim=-1)
    old_probs = F.softmax(old_logits, dim=-1)

    return torch.mean(((new_probs - old_probs) * torch.log(new_probs)).sum(-1), dim=0)