# run_ablation_shunyata.py -- Maya-Shunyata Paper 8 ablation study
# Split-CIFAR-100 CIL, 6 conditions
#
# A: P7 Maya-Manas baseline      -- no Karma, no pruning
# B: Karma + Vairagya only       -- no Viveka, no Chitta, no Manas-GANE
# C: Full affective + continuous pruning -- batch-level, expected instability
# D: Full affective + boundary pruning  -- canonical starred
# E: Full affective + aggressive threshold -- spike starvation risk
# F: Karma replaces Chitta entirely      -- full Chitta disable
#
# Canary: MayaNexusVS2026NLL_Bengaluru_Narasimha

import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch
import torch.nn as nn
from tqdm import tqdm

from maya_cl.utils.config import (
    EPOCHS_PER_TASK, NUM_TASKS, T_STEPS,
    VAIRAGYA_PROTECTION_THRESHOLD,
    REPLAY_BUFFER_SIZE, REPLAY_RATIO,
    REPLAY_VAIRAGYA_PARTIAL_LIFT,
    CIL_BOUNDARY_DECAY, BATCH_SIZE,
    REPLAY_PAIN_EXEMPT, A_MANAS,
    KARMA_THRESHOLD, KARMA_THRESHOLD_LOW,
)
from maya_cl.utils.seed import set_seed
from maya_cl.encoding.poisson import PoissonEncoder
from maya_cl.network.backbone import MayaShunyataNet
from maya_cl.network.affective_state import AffectiveState
from maya_cl.benchmark.split_cifar100 import (
    get_task_loaders, get_all_test_loaders, TASK_CLASSES
)
from maya_cl.benchmark.task_sequence import TaskSequencer
from maya_cl.plasticity.lability import LabilityMatrix
from maya_cl.plasticity.vairagya_decay import VairagyadDecay
from maya_cl.plasticity.viveka import VivekaConsistency
from maya_cl.plasticity.chitta import ChittaSamskara
from maya_cl.plasticity.manas import ManasConsistency
from maya_cl.plasticity.karma import KarmaShunyata
from maya_cl.eval.metrics import CLMetrics, evaluate_task
from maya_cl.eval.logger import RunLogger
from maya_cl.training.replay_buffer import ReplayBuffer

N_REPLAY = round(BATCH_SIZE * REPLAY_RATIO / (1.0 - REPLAY_RATIO))

CONDITIONS = {
    'A_baseline': {
        'use_karma':        False,
        'use_chitta':       True,
        'use_viveka':       True,
        'use_manas_gane':   True,
        'boundary_only':    True,
        'karma_threshold':  KARMA_THRESHOLD,
        'description':      'P7 Maya-Manas baseline — no Karma, no pruning',
    },
    'B_karma_vairagya': {
        'use_karma':        True,
        'use_chitta':       False,
        'use_viveka':       False,
        'use_manas_gane':   False,
        'boundary_only':    True,
        'karma_threshold':  KARMA_THRESHOLD,
        'description':      'Karma + Vairagya only — no higher affective dims',
    },
    'C_continuous_pruning': {
        'use_karma':        True,
        'use_chitta':       True,
        'use_viveka':       True,
        'use_manas_gane':   True,
        'boundary_only':    False,
        'karma_threshold':  KARMA_THRESHOLD,
        'description':      'Full affective + continuous batch-level pruning',
    },
    'D_boundary_canonical': {
        'use_karma':        True,
        'use_chitta':       True,
        'use_viveka':       True,
        'use_manas_gane':   True,
        'boundary_only':    True,
        'karma_threshold':  KARMA_THRESHOLD,
        'description':      'Full affective + boundary pruning — canonical',
    },
    'E_aggressive_threshold': {
        'use_karma':        True,
        'use_chitta':       True,
        'use_viveka':       True,
        'use_manas_gane':   True,
        'boundary_only':    True,
        'karma_threshold':  KARMA_THRESHOLD_LOW,
        'description':      f'Full affective + aggressive threshold={KARMA_THRESHOLD_LOW}',
    },
    'F_karma_replaces_chitta': {
        'use_karma':        True,
        'use_chitta':       False,
        'use_viveka':       True,
        'use_manas_gane':   True,
        'boundary_only':    True,
        'karma_threshold':  KARMA_THRESHOLD,
        'description':      'Karma replaces Chitta entirely — full Chitta disable',
    },
}


def run_condition(condition_name: str, seed: int = 42) -> dict:
    print("MayaNexusVS2026NLL_Bengaluru_Narasimha")
    assert condition_name in CONDITIONS
    cfg = CONDITIONS[condition_name]
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  Condition : {condition_name}")
    print(f"  {cfg['description']}")
    print(f"  Seed      : {seed}")
    print(f"{'='*60}")

    model         = MayaShunyataNet(use_orthogonal_head=False, a_manas=A_MANAS).to(device)
    encoder       = PoissonEncoder(T_STEPS)
    criterion     = nn.CrossEntropyLoss()
    optimizer     = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    affect        = AffectiveState(device)
    sequencer     = TaskSequencer()
    metrics       = CLMetrics(NUM_TASKS)
    logger        = RunLogger(f"ablation_{condition_name}")
    test_loaders  = get_all_test_loaders()
    replay_buffer = ReplayBuffer(max_per_class=REPLAY_BUFFER_SIZE)

    fc1_shape  = (model.fc1.fc.weight.shape[0], model.fc1.fc.weight.shape[1])
    fout_shape = (model.fc_out.weight.shape[0], model.fc_out.weight.shape[1])

    lability_fc1  = LabilityMatrix(fc1_shape,  device)
    vairagya_fc1  = VairagyadDecay(fc1_shape,  device)
    vairagya_fout = VairagyadDecay(fout_shape, device)
    viveka        = VivekaConsistency(fc1_shape, device)
    chitta        = ChittaSamskara(fc1_shape, device)
    manas_cons    = ManasConsistency(fc1_shape, device)
    karma         = KarmaShunyata(
        fc1_shape, device, threshold=cfg['karma_threshold'])

    w_prev     = model.fc1.fc.weight.data.clone()
    prev_loss  = None
    tasks_seen = 0

    for task_id in range(NUM_TASKS):
        train_loader, _ = get_task_loaders(task_id)
        sequencer.current_task = task_id
        current_classes = TASK_CLASSES[task_id]

        seen_classes = []
        for t in range(task_id + 1):
            seen_classes.extend(TASK_CLASSES[t])

        seen_mask = torch.zeros(fout_shape[0], dtype=torch.bool, device=device)
        for c in seen_classes:
            seen_mask[c] = True

        if task_id > 0:
            with torch.no_grad():
                vairagya_fc1.scores  *= CIL_BOUNDARY_DECAY
                vairagya_fout.scores *= CIL_BOUNDARY_DECAY

            if cfg['use_chitta']:
                moha_mask = chitta.detect_moha()
                if moha_mask.any():
                    chitta.apply_moha_release(moha_mask)
                chitta.on_task_boundary()

            if cfg['use_viveka']:
                viveka.on_task_boundary()

            affect.reset_experience()

            if cfg['use_karma'] and cfg['boundary_only']:
                buddhi_val = affect.buddhi_value()
                n_pruned   = karma.on_task_boundary(
                    model.fc1.fc.weight.data, buddhi=buddhi_val)
                if n_pruned > 0:
                    affect.update_shunyata(
                        n_pruned, fc1_shape[0] * fc1_shape[1])

            tasks_seen += 1

        print(f"--- Task {task_id} ---")

        for epoch in range(EPOCHS_PER_TASK):
            model.train()
            epoch_loss = 0.0

            for batch_idx, (images, labels) in enumerate(tqdm(
                    train_loader, desc=f"  Epoch {epoch+1}/{EPOCHS_PER_TASK}")):

                is_replay_batch = replay_buffer.is_ready() and task_id > 0
                if is_replay_batch:
                    r_imgs, r_lbls = replay_buffer.sample(N_REPLAY, device)
                    if r_imgs is not None:
                        images = torch.cat([images.to(device), r_imgs])
                        labels = torch.cat([labels.to(device), r_lbls])
                    else:
                        images = images.to(device)
                        labels = labels.to(device)
                        is_replay_batch = False
                else:
                    images = images.to(device)
                    labels = labels.to(device)

                spike_seq = encoder(images)
                model.reset()
                logits = model(spike_seq)

                peak_active = model.get_fc1_peak_active()

                with torch.no_grad():
                    v = model.fc1.lif.v
                    if v is not None and v.numel() > 0:
                        v_flat     = v.reshape(-1, fc1_shape[0])
                        post_mean  = v_flat.mean(dim=0)
                        active_fc1 = post_mean.unsqueeze(1).expand(fc1_shape) > 0.05
                    else:
                        active_fc1 = torch.zeros(
                            fc1_shape, dtype=torch.bool, device=device)

                    if cfg['use_manas_gane'] and cfg['use_viveka']:
                        peak_active_expanded = peak_active.unsqueeze(1).expand(fc1_shape)
                        manas_gane_mask = manas_cons.compute_manas_gane_mask(
                            viveka.scores, viveka_threshold=0.3) & peak_active_expanded
                    else:
                        manas_gane_mask = torch.zeros(
                            fc1_shape, dtype=torch.bool, device=device)

                loss = criterion(logits, labels)
                optimizer.zero_grad()
                loss.backward()

                retrograde_fired = False
                with torch.no_grad():
                    if model.fc_out.weight.grad is not None:
                        protected_fout = (
                            vairagya_fout.scores >= VAIRAGYA_PROTECTION_THRESHOLD
                        ).clone()
                        for c in current_classes:
                            protected_fout[c, :] = False
                        model.fc_out.weight.grad[protected_fout] = 0.0

                    if model.fc1.fc.weight.grad is not None:
                        protected_fc1 = (
                            vairagya_fc1.scores >= VAIRAGYA_PROTECTION_THRESHOLD
                        ).clone()
                        class_weights = model.fc_out.weight[current_classes, :]
                        cw_mean       = class_weights.abs().mean(dim=0)
                        threshold_80  = torch.quantile(cw_mean, 0.80)
                        important_fc1 = cw_mean > threshold_80
                        protected_fc1[important_fc1, :] = False

                        if is_replay_batch:
                            model.fc1.fc.weight.grad[protected_fc1] *= (
                                1.0 - REPLAY_VAIRAGYA_PARTIAL_LIFT)
                        else:
                            model.fc1.fc.weight.grad[protected_fc1] = 0.0

                    if cfg['use_chitta']:
                        chitta_gate      = chitta.compute_gradient_gate(
                            active_fc1, tasks_seen)
                        retrograde_fired = bool((chitta_gate < 1.0).any().item())
                        if retrograde_fired and model.fc1.fc.weight.grad is not None:
                            chitta.apply_gradient_gate(
                                model.fc1.fc.weight.grad, chitta_gate)
                            affect.update_chitta(
                                True, float((1.0 - chitta_gate).mean().item()))

                optimizer.step()

                with torch.no_grad():
                    if cfg['use_karma']:
                        w_current = model.fc1.fc.weight.data
                        karma.accumulate(w_current, w_prev)

                        # continuous pruning -- Condition C only
                        if not cfg['boundary_only']:
                            buddhi_val = affect.buddhi_value()
                            n_pruned   = karma.on_task_boundary(
                                model.fc1.fc.weight.data, buddhi=buddhi_val)
                            if n_pruned > 0:
                                affect.update_shunyata(
                                    n_pruned, fc1_shape[0] * fc1_shape[1])

                        karma.apply_mask(model.fc1.fc.weight.data)
                        w_prev = model.fc1.fc.weight.data.clone()

                epoch_loss += loss.item()

                with torch.no_grad():
                    cur_loss = loss.item()
                    conf     = sequencer.update_confidence(logits)

                    if REPLAY_PAIN_EXEMPT and is_replay_batch:
                        pain = False
                    else:
                        pain = sequencer.check_pain_signal(cur_loss, prev_loss, conf)
                        prev_loss = cur_loss

                    spike_rate = active_fc1.float().mean().item()
                    affect.update(conf, pain, spike_rate)
                    affect.update_manas(peak_active)

                    bhaya_val  = affect.bhaya.item()
                    buddhi_val = affect.buddhi_value()

                    if cfg['use_viveka']:
                        viveka_gain = viveka.compute_gain(
                            active_fc1, affect.viveka_signal(), tasks_seen)
                        viveka.update(active_fc1)
                    else:
                        viveka_gain = torch.ones(fc1_shape, device=device)

                    pain_fc1 = active_fc1 if pain else torch.zeros(
                        fc1_shape, dtype=torch.bool, device=device)
                    if pain:
                        lability_fc1.inject_pain(active_fc1)
                    lability_fc1.decay()

                    if cfg['use_chitta']:
                        chitta.update(active_fc1)

                    manas_viveka_gain = viveka_gain.clone()
                    if cfg['use_manas_gane'] and manas_gane_mask.any():
                        manas_viveka_gain[manas_gane_mask] *= 2.0

                    vairagya_fc1.accumulate(
                        active_fc1, pain_fc1,
                        bhaya=bhaya_val, buddhi=buddhi_val,
                        viveka_gain=manas_viveka_gain)
                    vairagya_fc1.apply_decay(model.fc1.fc.weight.data)

                    logit_mag   = logits.detach().abs().mean(dim=0)
                    active_fout = (
                        logit_mag.unsqueeze(1).expand(fout_shape) > logit_mag.mean()
                    )
                    active_fout = active_fout & seen_mask.unsqueeze(1)
                    pain_fout   = active_fout if pain else torch.zeros(
                        fout_shape, dtype=torch.bool, device=device)
                    vairagya_fout.accumulate(
                        active_fout, pain_fout,
                        bhaya=bhaya_val, buddhi=buddhi_val)

                    if cfg['use_manas_gane'] and peak_active is not None:
                        peak_active_expanded_bool = peak_active.unsqueeze(1).expand(fc1_shape)
                        manas_peak_fc1 = active_fc1 & peak_active_expanded_bool
                        manas_cons.update(manas_peak_fc1)

                manas_peak_fraction = float(peak_active.float().mean().item())

                logger.log_batch(
                    task=task_id, epoch=epoch, batch=batch_idx,
                    loss=cur_loss, confidence=conf, pain_fired=pain,
                    lability_mean=lability_fc1.get().mean().item(),
                    vairagya_protection=vairagya_fc1.protection_fraction(),
                    affective=affect.as_dict(),
                    samskara_mean=chitta.mean_samskara() if cfg['use_chitta'] else 0.0,
                    moha_fraction=chitta.moha_fraction() if cfg['use_chitta'] else 0.0,
                    retrograde_fired=bool(retrograde_fired),
                    manas_peak_fraction=manas_peak_fraction,
                    karma_mean=karma.karma_mean() if cfg['use_karma'] else 0.0,
                    pruned_fraction=karma.pruned_fraction() if cfg['use_karma'] else 0.0,
                    shunyata_events=karma.total_pruned() if cfg['use_karma'] else 0,
                )

            with torch.no_grad():
                for buf_imgs, buf_lbls in train_loader:
                    replay_buffer.update(buf_imgs, buf_lbls)
                    break

            print(f"    Loss: {epoch_loss/len(train_loader):.4f} | "
                  f"Bhaya: {affect.bhaya.item():.3f} | "
                  f"Buddhi: {affect.buddhi_value():.3f} | "
                  f"Karma: {karma.karma_mean():.4f} | "
                  f"Pruned: {karma.pruned_fraction()*100:.2f}% | "
                  f"V-fc1: {vairagya_fc1.protection_fraction()*100:.1f}%")

        print(f"  Evaluating after Task {task_id} [CIL]...")
        acc_dict = {}
        for t in range(NUM_TASKS):
            acc = evaluate_task(
                model, test_loaders[t], device, encoder, T_STEPS,
                task_classes=None)
            metrics.update(trained_up_to=task_id, task_id=t, accuracy=acc)
            acc_dict[f"task_{t}"] = round(acc * 100, 2)
            print(f"    Task {t}: {acc*100:.2f}%")

        logger.log_task_summary(
            task_id, acc_dict, metrics.summary(),
            karma.summary() if cfg['use_karma'] else None)

    metrics.print_matrix()
    final = metrics.summary()
    print(f"\n{'='*60}")
    print(f"  Condition: {condition_name} | seed={seed}")
    print(f"  AA  : {final['AA']}%")
    print(f"  BWT : {final['BWT']}%")
    print(f"  FWT : {final['FWT']}%")
    if cfg['use_karma']:
        print(f"  Total pruned: {karma.total_pruned()} synapses "
              f"({karma.pruned_fraction()*100:.2f}%)")
    print(f"{'='*60}")
    logger.log_final(final)
    logger.close()
    return final


if __name__ == "__main__":
    results = {}
    for cond in CONDITIONS:
        results[cond] = run_condition(cond, seed=42)

    print(f"\n{'='*60}")
    print("ABLATION SUMMARY -- Maya-Shunyata Paper 8")
    print(f"{'='*60}")
    for cond, r in results.items():
        print(f"  {cond:30s} AA={r['AA']}% BWT={r['BWT']}%")
    print(f"{'='*60}")
