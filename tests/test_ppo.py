import math

import torch

from glimmer_ppo.ppo import explained_variance, gae, kl_penalty, policy_loss, value_loss


def test_gae_matches_plan_doc_example():
    # 3 decisions, rewards 0,0,1, values 0.2,0.4,0.7, gamma 1, lambda 0.95 (worked example in the plan doc)
    adv, ret = gae([0, 0, 1], [0.2, 0.4, 0.7], [False, False, True], gamma=1.0, lam=0.95)
    assert torch.allclose(adv, torch.tensor([0.75575, 0.585, 0.3]), atol=1e-5)
    assert torch.allclose(ret, torch.tensor([0.95575, 0.985, 1.0]), atol=1e-5)


def test_gae_does_not_bootstrap_across_episodes():
    # episode A ends at t=0; episode B's big value must not leak back into A
    adv, _ = gae([1.0, 0.0], [0.0, 10.0], [True, False], gamma=1.0, lam=1.0, last_value=0.0)
    assert adv[0].item() == 1.0
    assert adv[1].item() == -10.0


def test_gae_bootstraps_a_batch_cut_mid_episode():
    adv, _ = gae([0.0], [0.5], [False], gamma=1.0, lam=0.95, last_value=0.8)
    assert math.isclose(adv[0].item(), 0.3, rel_tol=1e-6)


def test_clip_stops_gradient_past_the_trust_region():
    # A = 0.756, eps = 0.2, ratio 1.3 -> objective uses 1.2 * A and the token gets no gradient
    logp_old = torch.tensor([0.0])
    logp = torch.tensor([math.log(1.3)], requires_grad=True)
    loss, stats = policy_loss(logp, logp_old, torch.tensor(0.756), clip=0.2)
    assert math.isclose(-loss.item(), 1.2 * 0.756, rel_tol=1e-6)
    loss.backward()
    assert logp.grad.abs().item() == 0.0
    assert stats['clipfrac'] == 1.0


def test_unclipped_ratio_gets_gradient_toward_positive_advantage():
    logp = torch.tensor([0.0, 0.0], requires_grad=True)
    loss, stats = policy_loss(logp, torch.zeros(2), torch.tensor(1.0), clip=0.2)
    loss.backward()
    assert (logp.grad < 0).all()          # minimizing loss raises log-probs
    assert stats['approx_kl'] == 0.0


def test_negative_advantage_is_clipped_on_the_low_side():
    logp = torch.tensor([math.log(0.7)], requires_grad=True)
    loss, _ = policy_loss(logp, torch.tensor([0.0]), torch.tensor(-1.0), clip=0.2)
    assert math.isclose(loss.item(), 0.8, rel_tol=1e-6)   # min(0.7*-1, 0.8*-1) = -0.8
    loss.backward()
    assert logp.grad.item() == 0.0


def test_value_loss_and_clipping():
    v = torch.tensor(1.0)
    assert value_loss(v, torch.tensor(1.0), torch.tensor(0.0), None).item() == 0.5
    # clipped value 0.2 is farther from target 1.0 than 1.0 itself, so max picks the clipped error
    assert math.isclose(value_loss(torch.tensor(1.0), torch.tensor(0.0), torch.tensor(1.0), 0.2).item(),
                        0.5 * 0.8 ** 2, rel_tol=1e-6)


def test_kl_and_explained_variance():
    assert math.isclose(kl_penalty(torch.tensor([-1.0, -2.0]), torch.tensor([-1.5, -2.0])).item(), 0.5)
    assert explained_variance([1, 2, 3], [1, 2, 3]) == 1.0
    assert explained_variance([0, 0, 0], [1, 2, 3]) == 0.0
