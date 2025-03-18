import copy
from components.episode_buffer import EpisodeBatch
from modules.critics.facmac import FACMACDiscreteCritic, PeVFA_FACMACDiscreteCritic
# from components.action_selectors import multinomial_entropy
import torch as th
from torch.optim import RMSprop, Adam
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer, PeVFA_QMixer, V_Net
from modules.mixers.qmix_ablations import VDNState, QMixerNonmonotonic
from modules.mixers.graphmix import GraphMixer
from utils.rl_utils import build_td_lambda_targets
from modules.graphlearner.GTSModel import GTSModel
from components.attention_module import AttentionModule
from components.gcn_module import GCNModule
import random
import torch.nn as nn
import time
import math
from torch.nn import functional as F
import torch

import numpy as np


class EA_FACMACDiscreteLearner:
    def __init__(self, mac, scheme, logger, args):
        if args.LTSCG:
            self.temperature = 0.5
            self._model_kwargs = args.gtsmodel
            self._model_kwargs['input_dim'] = args.obs_shape
            self._model_kwargs['output_dim'] = args.obs_shape
            self._model_kwargs['num_nodes'] = args.n_agents
            self._model_kwargs['dim_fc'] = args.obs_shape * (args.episode_limit + 1)
            graph_learner = GTSModel(self.temperature, logger, self._model_kwargs)
            self.graph_learner = graph_learner.cuda() if th.cuda.is_available() else graph_learner
            input_shape = args.obs_shape + 1
            ################## for state #######################
            self.n_gcn_layers = args.number_gcn_layers
            self.graph_emb_dim = args.state_shape
            self.graph_emb_hid = args.graph_emb_hid
            self.graph_input_obs_MLP = self._mlp(input_shape, self.graph_emb_hid, self.graph_emb_dim)
            self.attention_layer = AttentionModule((self.graph_emb_dim), attention_type='general')
            self.gcn_layers = nn.ModuleList(
                [GCNModule(in_features=(self.graph_emb_dim), out_features=(self.graph_emb_dim), bias=True, id=i) for i
                 in
                 range(self.n_gcn_layers)])

            self.mlp_emb_hid = args.mlp_emb_hid
            self.mlp_out = args.mlp_out
            self.graph_output_MLP = self._mlp(self.graph_emb_dim, self.mlp_emb_hid, self.mlp_out)
            self.state_MLP = self._mlp(args.state_shape, self.mlp_emb_hid, self.mlp_out)
            ################## for obs #######################
            self.graph_obs_MLP = self._mlp(args.obs_shape, self.mlp_emb_hid, self.mlp_out)
            self.next_obs_MLP = self._mlp(args.obs_shape, self.mlp_emb_hid, self.mlp_out)

            if th.cuda.is_available():
                self.graph_input_obs_MLP.cuda()
                self.attention_layer.cuda()
                self.gcn_layers.cuda()
                self.graph_output_MLP.cuda()
                self.state_MLP.cuda()

                self.graph_obs_MLP.cuda()
                self.next_obs_MLP.cuda()

        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger

        self.device = th.device('cuda' if args.use_cuda else 'cpu')

        self.mac = mac
        self.target_mac = copy.deepcopy(self.mac)

        self.agent_params = list(mac.parameters())

        self.critic = FACMACDiscreteCritic(scheme, args)

        self.target_critic = copy.deepcopy(self.critic)
        self.critic_params = list(self.critic.parameters())

        if args.EA:
            self.PeVFA_critic = PeVFA_FACMACDiscreteCritic(scheme, args)
            self.PeVFA_params = list(self.PeVFA_critic.parameters())
            self.target_PeVFA_critic = copy.deepcopy(self.PeVFA_critic)

            if args.mixer == "graph":
                self.PeVFA_mixer = GraphMixer(args)
            else:
                self.PeVFA_mixer = PeVFA_QMixer(args)

            self.PeVFA_params += list(self.PeVFA_mixer.parameters())
            self.target_PeVFA_mixer = copy.deepcopy(self.PeVFA_mixer)
            self.PeVFA_optimiser = Adam(params=self.PeVFA_params, lr=args.critic_lr,
                                        eps=getattr(args, "optimizer_epsilon", 10E-8))

        self.mixer = None
        if args.mixer is not None and self.args.n_agents > 1:  # if just 1 agent do not mix anything
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            elif args.mixer == "vdn-s":
                self.mixer = VDNState(args)
            elif args.mixer == "qmix-nonmonotonic":
                self.mixer = QMixerNonmonotonic(args)
            elif args.mixer == "graph":
                self.mixer = GraphMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))

            self.critic_params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        if getattr(self.args, "optimizer", "rmsprop") == "rmsprop":
            self.agent_optimiser = RMSprop(params=self.agent_params, lr=args.lr, alpha=args.optim_alpha,
                                           eps=args.optim_eps)
        elif getattr(self.args, "optimizer", "rmsprop") == "adam":
            self.agent_optimiser = Adam(params=self.agent_params, lr=args.lr,
                                        eps=getattr(args, "optimizer_epsilon", 10E-8))
        else:
            raise Exception("unknown optimizer {}".format(getattr(self.args, "optimizer", "rmsprop")))

        if getattr(self.args, "optimizer", "rmsprop") == "rmsprop":
            self.critic_optimiser = RMSprop(params=self.critic_params, lr=args.critic_lr, alpha=args.optim_alpha,
                                            eps=args.optim_eps)
        elif getattr(self.args, "optimizer", "rmsprop") == "adam":
            self.critic_optimiser = Adam(params=self.critic_params, lr=args.critic_lr,
                                         eps=getattr(args, "optimizer_epsilon", 10E-8))
        else:
            raise Exception("unknown optimizer {}".format(getattr(self.args, "optimizer", "rmsprop")))

        self.log_stats_t = -self.args.learner_log_interval - 1
        self.last_target_update_episode = 0
        self.critic_training_steps = 0

    def train(self, episode_sample: EpisodeBatch, all_teams, max_ep_t, t_env: int,
              episode_num: int):
        # Get the relevant quantities
        if self.args.LTSCG:
            batch = episode_sample[:, :max_ep_t]
            rewards = batch['reward'][:, :-1]
            actions = batch['actions_onehot'][:, :]
            terminated = batch['terminated'].float()
            mask = batch['filled'].float()
            mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
            avail_actions = batch['avail_actions'][:, :-1]
            temp_mask = mask

            obs = episode_sample['obs'][:, :]
            obs = obs.permute(2, 0, 1, 3)
            episode_obs = obs.reshape(self.args.n_agents, batch.batch_size, -1)
            input_obs = batch['obs'][:, :-1].transpose(1, 0)
            input_last_a = batch['actions'][:, :-1].transpose(1, 0)

            obs_with_action = th.cat([input_obs, input_last_a], dim=3)
            encoder_input = obs_with_action.reshape(max_ep_t - 1, batch.batch_size, -1)
            graph = batch['graph'][:, :]
            self.graph_learner = self.graph_learner.train()
            gumbel_adj, graph_decoder_out = self.graph_learner(1, encoder_input, episode_obs, graph, self.temperature,
                                                               1, 1,
                                                               1)

            graph_decoder_out = graph_decoder_out.reshape(max_ep_t - 1, batch.batch_size, self.args.n_agents, -1)

            learn_graph = th.unsqueeze(gumbel_adj, dim=0)
            learn_graph_batch = learn_graph.repeat(batch.batch_size, 1, 1)
            batch['graph'][:, :] = learn_graph_batch

        else:
            batch = episode_sample
            rewards = batch["reward"][:, :-1]
            # actions = batch["actions"][:, :]
            actions = batch["actions_onehot"][:, :]
            terminated = batch["terminated"].float()
            mask = batch["filled"].float()
            mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
            avail_actions = batch["avail_actions"][:, :-1]
            temp_mask = mask

        if self.args.LTSCG:
            ############### all obs and state loss ##############
            graph_embeddings_collection = []
            graph_embeddings_0 = self.graph_input_obs_MLP.forward(obs_with_action)
            graph_embeddings_collection.append(graph_embeddings_0)
            attention_weights = self.attention_layer.forward(graph_embeddings_0)
            for i_layer, gcn_layer in enumerate(self.gcn_layers):
                embeddings_gcn = gcn_layer.forward(graph_embeddings_collection[i_layer],
                                                   learn_graph_batch.unsqueeze(0) * attention_weights)
                graph_embeddings_collection.append(embeddings_gcn)

            avg_pool = th.mean((graph_embeddings_collection[-1]), dim=2).transpose(1, 0)

            graph_obs_emb = self.graph_output_MLP.forward(avg_pool)
            state_emb = self.state_MLP(batch['state'][:, :-1])
            g_loss = nn.MSELoss()
            gstate_loss = g_loss(graph_obs_emb, state_emb)

            ############### obs t and next obs loss ##############
            graph_obs_emb = self.graph_obs_MLP.forward(input_obs + graph_decoder_out)
            next_obs_emb = self.next_obs_MLP.forward(batch["obs"][:, 1:].transpose(1, 0))
            gobs_loss = g_loss(graph_obs_emb, next_obs_emb)

        if self.args.use_cuda:
            self.mac.cuda()
            self.target_mac.cuda()
            self.critic.cuda()
            self.target_critic.cuda()
            if self.mixer is not None:
                self.mixer.cuda()
                self.target_mixer.cuda()

        if self.args.EA:
            # Train the critic batched
            index = random.sample(list(range(self.args.pop_size + 1)), 1)[0]
            selected_team = all_teams[index]

            if self.args.use_cuda:
                selected_team.cuda()
                self.PeVFA_mixer.cuda()
                self.target_PeVFA_mixer.cuda()
                self.target_PeVFA_critic.cuda()
                self.PeVFA_critic.cuda()

            target_mac_out = []
            selected_team.init_hidden(batch.batch_size)
            hidden_states = []
            for t in range(batch.max_seq_length):
                target_act_outs = selected_team.select_actions(batch, t_ep=t, t_env=t_env, test_mode=True)
                target_mac_out.append(target_act_outs)
                hidden_states.append(
                    selected_team.hidden_states.view(batch.batch_size, self.args.n_agents, -1))

            target_mac_out = th.stack(target_mac_out, dim=1)  # Concat over time

            hidden_states = th.stack(hidden_states, dim=1)

            if self.args.SameMixer:
                q_taken_origi_ea, _ = self.critic(batch["obs"][:, :-1], actions[:, :-1])
                if self.mixer is not None:
                    if self.args.mixer == "vdn":
                        assert 1 == 2
                    elif self.args.mixer == "graph":
                        q_taken, local_rewards, alive_agents_mask = self.mixer(q_taken_origi_ea,
                                                                               batch["state"][:, :-1],
                                                                               agent_obs=batch["obs"][:, :-1],
                                                                               team_rewards=rewards,
                                                                               hidden_states=hidden_states[:,
                                                                                             :-1],
                                                                               )
                    else:
                        q_taken = self.mixer.forward(q_taken_origi_ea.view(batch.batch_size, -1, 1),
                                                     batch["state"][:, :-1])

                target_vals, _ = self.target_critic(batch["obs"][:, :], target_mac_out.detach())
                if self.mixer is not None:
                    if self.args.mixer == "graph":
                        target_vals = self.target_mixer(target_vals.view(batch.batch_size, -1, 1),
                                                        batch["state"][:, :],
                                                        agent_obs=batch["obs"][:, :],
                                                        hidden_states=hidden_states
                                                        )[0]
                    else:
                        target_vals = self.target_mixer.forward(target_vals.view(batch.batch_size, -1, 1),
                                                                batch["state"][:, :], )
            else:
                q_taken_origi_ea, _ = self.PeVFA_critic(batch["obs"][:, :-1], actions[:, :-1])
                if self.PeVFA_mixer is not None:
                    if self.args.mixer == "vdn":
                        assert 1 == 2
                    elif self.args.mixer == "graph":
                        q_taken, local_rewards, alive_agents_mask = self.PeVFA_mixer(q_taken_origi_ea,
                                                                                     batch["state"][:, :-1],
                                                                                     agent_obs=batch["obs"][:, :-1],
                                                                                     team_rewards=rewards,
                                                                                     hidden_states=hidden_states[:,
                                                                                                   :-1],
                                                                                     )
                    else:
                        q_taken = self.PeVFA_mixer.forward(q_taken_origi_ea.view(batch.batch_size, -1, 1),
                                                           batch["state"][:, :-1])

                target_vals, _ = self.target_PeVFA_critic(batch["obs"][:, :], target_mac_out.detach())
                if self.mixer is not None:
                    if self.args.mixer == "graph":
                        target_vals = self.target_PeVFA_mixer(target_vals.view(batch.batch_size, -1, 1),
                                                              batch["state"][:, :],
                                                              agent_obs=batch["obs"][:, :],
                                                              hidden_states=hidden_states
                                                              )[0]
                    else:
                        target_vals = self.target_PeVFA_mixer.forward(target_vals.view(batch.batch_size, -1, 1),
                                                                      batch["state"][:, :], )

            if self.mixer is not None:
                q_taken = q_taken.view(batch.batch_size, -1, 1)
                target_vals = target_vals.view(batch.batch_size, -1, 1)
            else:
                q_taken = q_taken.view(batch.batch_size, -1, self.n_agents)
                target_vals = target_vals.view(batch.batch_size, -1, self.n_agents)

            targets = build_td_lambda_targets(batch["reward"], terminated, mask, target_vals,
                                              self.n_agents,
                                              self.args.gamma, self.args.td_lambda)
            mask = temp_mask[:, :-1]
            td_error = (q_taken - targets.detach())
            mask = mask.expand_as(td_error)
            masked_td_error = td_error * mask
            ea_loss = (masked_td_error ** 2).sum() / mask.sum()

            if self.args.LTSCG:
                ea_loss = ea_loss + gobs_loss + gstate_loss
            elif self.args.mixer == "graph" and self.args.graph_loss == True:
                local_targets = local_rewards.detach() + self.args.gamma * (1 - terminated[:, :-1]).repeat(1, 1,
                                                                                                           self.args.n_agents)

                # Td-error
                local_td_error = (q_taken_origi_ea.reshape(local_targets.size()) - local_targets)
                local_mask = mask.repeat(1, 1, self.args.n_agents) * alive_agents_mask.float()

                # 0-out the targets that came from padded data
                local_masked_td_error = local_td_error * local_mask

                # Normal L2 loss, take mean over actual data
                local_loss = (local_masked_td_error ** 2).sum() / mask.sum()

                # ea_loss += local_loss

            if self.args.SameMixer:
                self.critic_optimiser.zero_grad()
                ea_loss.backward()
                ea_critic_grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
                self.critic_optimiser.step()
            else:
                self.PeVFA_optimiser.zero_grad()
                ea_loss.backward()
                ea_critic_grad_norm = th.nn.utils.clip_grad_norm_(self.PeVFA_params, self.args.grad_norm_clip)
                self.PeVFA_optimiser.step()

        # print("1 ", time.time()-start)
        start = time.time()

        target_mac_out = []
        target_hidden_states = []
        hidden_states = []
        self.target_mac.init_hidden(batch.batch_size)
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_act_outs = self.target_mac.select_actions(batch, t_ep=t, t_env=t_env, test_mode=True)
            target_mac_out.append(target_act_outs)
            target_hidden_states.append(
                self.target_mac.hidden_states.view(batch.batch_size, self.args.n_agents, -1))
            hidden_states.append(self.mac.hidden_states.view(batch.batch_size, self.args.n_agents, -1))

        target_mac_out = th.stack(target_mac_out, dim=1)  # Concat over time
        target_hidden_states = th.stack(target_hidden_states, dim=1)
        hidden_states = th.stack(hidden_states, dim=1)

        q_taken_origi, _ = self.critic(batch["obs"][:, :-1], actions[:, :-1])
        if self.mixer is not None:
            if self.args.mixer == "graph":
                q_taken, local_rewards, alive_agents_mask = self.mixer(q_taken_origi,
                                                                       batch["state"][:, :-1],
                                                                       agent_obs=batch["obs"][:, :-1],
                                                                       team_rewards=rewards,
                                                                       hidden_states=hidden_states[:, :-1],
                                                                       )
            else:
                q_taken = self.mixer(q_taken_origi.view(batch.batch_size, -1, 1), batch["state"][:, :-1])

        target_vals, _ = self.target_critic(batch["obs"][:, :], target_mac_out.detach())
        if self.mixer is not None:
            if self.args.mixer == "graph":
                target_vals = self.target_mixer(target_vals.view(batch.batch_size, -1, 1),
                                                batch["state"][:, :],
                                                agent_obs=batch["obs"][:, :],
                                                hidden_states=hidden_states
                                                )[0]
            else:
                target_vals = self.target_mixer(target_vals.view(batch.batch_size, -1, 1), batch["state"][:, :])

        if self.mixer is not None:
            q_taken = q_taken.view(batch.batch_size, -1, 1)
            target_vals = target_vals.view(batch.batch_size, -1, 1)
        else:
            q_taken = q_taken.view(batch.batch_size, -1, self.n_agents)
            target_vals = target_vals.view(batch.batch_size, -1, self.n_agents)

        targets_1 = build_td_lambda_targets(batch["reward"], terminated, mask, target_vals,
                                            self.n_agents,
                                            self.args.gamma, self.args.td_lambda)
        mask = temp_mask[:, :-1]
        td_error = (q_taken - targets_1.detach())
        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask

        loss = (masked_td_error ** 2).sum() / mask.sum()

        if self.args.mixer == "graph" and self.args.graph_loss == True:
            local_targets = local_rewards.detach() + self.args.gamma * (1 - terminated[:, :-1]).repeat(1, 1,
                                                                                                       self.args.n_agents)

            # Td-error
            local_td_error = (q_taken_origi.reshape(local_targets.size()) - local_targets)
            local_mask = mask.repeat(1, 1, self.args.n_agents) * alive_agents_mask.float()

            # 0-out the targets that came from padded data
            local_masked_td_error = local_td_error * local_mask

            # Normal L2 loss, take mean over actual data
            local_loss = (local_masked_td_error ** 2).sum() / mask.sum()

            # loss += local_loss

        self.critic_optimiser.zero_grad()
        loss.backward()
        critic_grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
        self.critic_optimiser.step()
        self.critic_training_steps += 1

        # print("2 ", time.time() - start)
        start = time.time()

        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            act_outs = self.mac.select_actions(batch, t_ep=t, t_env=t_env, test_mode=False, explore=False)
            mac_out.append(act_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        chosen_action_qvals, _ = self.critic(batch["obs"][:, :-1], mac_out)

        if self.mixer is not None:
            if self.args.mixer == "graph":
                chosen_action_qvals, _, _ = self.mixer(chosen_action_qvals,
                                                       batch["state"][:, :-1],
                                                       agent_obs=batch["obs"][:, :-1],
                                                       team_rewards=rewards,
                                                       hidden_states=hidden_states[:, :-1],
                                                       )
            else:
                chosen_action_qvals = self.mixer(chosen_action_qvals.view(batch.batch_size, -1, 1),
                                                 batch["state"][:, :-1])

        # Compute the actor loss
        pg_loss = - (chosen_action_qvals * mask).sum() / mask.sum()

        # print("3 ", time.time() - start)
        start = time.time()

        if self.args.EA:
            mac_out = []
            hidden_states = []
            index = random.sample(list(range(self.args.pop_size + 1)), 1)[0]
            selected_team = all_teams[index]
            selected_team.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length - 1):
                act_outs = selected_team.select_actions(batch, t_ep=t, t_env=t_env, test_mode=False,
                                                        explore=False)

                mac_out.append(act_outs)
                hidden_states.append(selected_team.hidden_states.view(batch.batch_size, self.args.n_agents, -1))

            mac_out = th.stack(mac_out, dim=1)  # Concat over time
            hidden_states = th.stack(hidden_states, dim=1)

            chosen_action_qvals, _ = self.critic(batch["obs"][:, :-1], mac_out)

            if self.args.SameMixer:
                if self.mixer is not None:
                    if self.args.mixer == "graph":
                        chosen_action_qvals, local_rewards, alive_agents_mask = self.mixer(
                            chosen_action_qvals.view(batch.batch_size, -1, 1),
                            batch["state"][:, :-1],
                            agent_obs=batch["obs"][:, :-1],
                            team_rewards=rewards,
                            hidden_states=hidden_states
                        )
                    else:
                        chosen_action_qvals = self.mixer.forward(
                            chosen_action_qvals.view(batch.batch_size, -1, 1),
                            batch["state"][:, :-1])
            else:
                if self.PeVFA_mixer is not None:
                    if self.args.mixer == "graph":
                        chosen_action_qvals, local_rewards, alive_agents_mask = self.PeVFA_mixer(
                            chosen_action_qvals.view(batch.batch_size, -1, 1),
                            batch["state"][:, :-1],
                            agent_obs=batch["obs"][:, :-1],
                            team_rewards=rewards,
                            hidden_states=hidden_states
                        )
                    else:
                        chosen_action_qvals = self.PeVFA_mixer.forward(
                            chosen_action_qvals.view(batch.batch_size, -1, 1),
                            batch["state"][:, :-1])

            # Compute the actor loss
            # ea_pg_loss = - self.args.EA_alpha * (
            #         chosen_action_qvals * mask).sum() / mask.sum() + self.args.state_alpha * MINE_loss
            ea_pg_loss = -  (
                    chosen_action_qvals * mask).sum() / mask.sum()
        else:
            ea_pg_loss = 0.0
        # ea_pg_loss = 0.0
        # total_loss = pg_loss + self.args.EA_alpha *ea_pg_loss

        total_loss = self.args.Org_alpha * pg_loss + ea_pg_loss

        if self.args.mixer == "graph" and self.args.graph_loss == True:
            local_targets = local_rewards.detach() + self.args.gamma * (1 - terminated[:, :-1]).repeat(1, 1,
                                                                                                       self.args.n_agents)

            # Td-error
            local_td_error = (q_taken_origi_ea.reshape(local_targets.size()) - local_targets)
            local_mask = mask.repeat(1, 1, self.args.n_agents) * alive_agents_mask.float()

            # 0-out the targets that came from padded data
            local_masked_td_error = local_td_error * local_mask

            # Normal L2 loss, take mean over actual data
            local_loss = (local_masked_td_error ** 2).sum() / mask.sum()

            # total_loss += local_loss

        # Optimise agents
        self.agent_optimiser.zero_grad()
        total_loss.backward()
        agent_grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.agent_optimiser.step()

        # print("3 ", time.time() - start)
        start = time.time()

        if getattr(self.args, "target_update_mode", "hard") == "hard":
            if (self.critic_training_steps - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
                self._update_targets()
                self.last_target_update_episode = self.critic_training_steps
        elif getattr(self.args, "target_update_mode", "hard") in ["soft", "exponential_moving_average"]:
            self._update_targets_soft(tau=getattr(self.args, "target_update_tau", 0.001))
        else:
            raise Exception(
                "unknown target update mode: {}!".format(getattr(self.args, "target_update_mode", "hard")))

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("critic_loss", loss.item(), t_env)
            self.logger.log_stat("critic_grad_norm", critic_grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", masked_td_error.abs().sum().item() / mask_elems, t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                                 t_env)
            self.log_stats_t = t_env

    def _update_targets_soft(self, tau):

        if self.args.EA:
            for target_param, param in zip(self.target_PeVFA_critic.parameters(), self.PeVFA_critic.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

            for target_param, param in zip(self.target_PeVFA_mixer.parameters(), self.PeVFA_mixer.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        for target_param, param in zip(self.target_mac.parameters(), self.mac.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        if self.mixer is not None:
            for target_param, param in zip(self.target_mixer.parameters(), self.mixer.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        if self.args.verbose:
            self.logger.console_logger.info("Updated all target networks (soft update tau={})".format(tau))

    def _update_targets(self):

        if self.args.EA:
            self.target_PeVFA_mixer.load_state_dict(self.PeVFA_mixer.state_dict())
            self.target_PeVFA_critic.load_state_dict(self.PeVFA_critic.state_dict())
        self.target_mac.load_state(self.mac)
        self.target_critic.load_state_dict(self.critic.state_dict())
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated all target networks")

    def cuda(self, device="cuda:0"):
        self.device = device
        self.mac.cuda(device=device)
        self.target_mac.cuda(device=device)
        self.critic.cuda(device=device)
        self.target_critic.cuda(device=device)
        if self.mixer is not None:
            self.mixer.cuda(device=device)
            self.target_mixer.cuda(device=device)

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.agent_optimiser.load_state_dict(
            th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))

    @staticmethod
    def _mlp(input, hidden_dims, output):
        """ Creates an MLP with the specified input and output dimensions and (optional) hidden layers. """
        hidden_dims = [] if hidden_dims is None else hidden_dims
        hidden_dims = [hidden_dims] if isinstance(hidden_dims, int) else hidden_dims
        dim = input
        layers = []
        for d in hidden_dims:
            layers.append(nn.Linear(dim, d))
            layers.append(nn.ReLU())
            dim = d

        layers.append(nn.Linear(dim, output))
        return (nn.Sequential)(*layers)
