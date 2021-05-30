import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Code comes from the PLE implementation of PaddlePaddle.
"""


class PLELayer(nn.Module):
    def __init__(self, feature_size, task_num, exp_per_task, shared_num,
                 expert_size, tower_size, level_number):
        super(PLELayer, self).__init__()

        self.task_num = task_num
        self.exp_per_task = exp_per_task
        self.shared_num = shared_num
        self.expert_size = expert_size
        self.tower_size = tower_size
        self.level_number = level_number

        # ple layer
        self.ple_layers = nn.ModuleDict()
        for i in range(0, self.level_number):
            name = 'lev_' + str(i)

            if i == self.level_number - 1:
                self.ple_layers.add_module(name, SinglePLELayer(
                    feature_size, task_num, exp_per_task, shared_num,
                    expert_size, name, True))

            else:
                self.ple_layers.add_module(name, SinglePLELayer(
                    feature_size, task_num, exp_per_task, shared_num,
                    expert_size, name, False))
                feature_size = expert_size

        # task tower
        self.tower = nn.ModuleDict()
        self.tower_out = nn.ModuleDict()

        for i in range(0, self.task_num):
            name = 'tower_' + str(i)
            self.tower.add_module(name, nn.Linear(
                expert_size,
                tower_size
            ))

            name = 'tower_out_' + str(i)
            self.tower_out.add_module(name, nn.Linear(
                tower_size,
                2
            ))

    def forward(self, input_data):
        inputs_ple = []

        # task_num part + shared part
        for i in range(0, self.task_num + 1):
            inputs_ple.append(input_data)

        # multiple ple layer
        ple_out = []
        keys = list(self.ple_layers.keys())
        for i in range(0, self.level_number):
            ple_out = self.ple_layers[keys[i]](inputs_ple)
            inputs_ple = ple_out

        # assert len(ple_out) == self.task_num
        output_layers = []
        keys = [list(self.tower.keys()), list(self.tower_out.keys())]
        for i in range(0, self.task_num):
            cur_tower = self.tower[keys[0][i]](ple_out[i])
            cur_tower = F.relu(cur_tower)

            out = self.tower_out[keys[1][i]](cur_tower)
            out = F.softmax(out, dim=-1)

            out = torch.clamp(out, min=1e-15, max=1.0 - 1e-15)
            output_layers.append(out)

        return output_layers


class SinglePLELayer(nn.Module):
    def __init__(self, input_feature_size, task_num, exp_per_task, shared_num,
                 expert_size, level_name, if_last):
        super(SinglePLELayer, self).__init__()

        self.task_num = task_num
        self.exp_per_task = exp_per_task
        self.shared_num = shared_num
        self.expert_size = expert_size
        self.if_last = if_last

        # task-specific expert part
        self.task_specific_expert = nn.ModuleDict()
        for i in range(0, self.task_num):
            for j in range(0, self.exp_per_task):
                name = level_name + "_exp_" + str(i) + "_" + str(j)
                self.task_specific_expert.add_module(name, nn.Linear(
                    input_feature_size,
                    expert_size
                ))

        # shared expert part
        self.shared_expert = nn.ModuleDict()
        for i in range(0, self.shared_num):
            name = level_name + "_exp_shared_" + str(i)
            self.shared_expert.add_module(name, nn.Linear(
                input_feature_size,
                expert_size
            ))

        # task gate part
        self.gate = nn.ModuleDict()
        cur_expert_num = self.exp_per_task + self.shared_num
        for i in range(0, self.task_num):
            name = level_name + "_gate_" + str(i)
            self.gate.add_module(name, nn.Linear(
                input_feature_size,
                cur_expert_num
            ))

        # shared gate part
        if not if_last:
            cur_expert_num = self.task_num * self.exp_per_task + self.shared_num
            self.gate_shared = nn.Linear(input_feature_size, cur_expert_num)

        # dont forget to init weights

    def forward(self, input_data):
        expert_outputs = []

        # task-specific expert part
        keys = list(self.task_specific_expert.keys())
        for i in range(0, self.task_num):
            for j in range(0, self.exp_per_task):
                linear_out = self.task_specific_expert[keys[i * self.exp_per_task + j]](
                    input_data[i])
                expert_output = F.relu(linear_out)
                expert_outputs.append(expert_output)

        # shared expert part
        keys = list(self.shared_expert.keys())
        for i in range(0, self.shared_num):
            linear_out = self.shared_expert[keys[i]](input_data[-1])
            expert_output = F.relu(linear_out)
            expert_outputs.append(expert_output)

        # task gate part
        outputs = []
        keys = list(self.gate.keys())
        for i in range(0, self.task_num):
            cur_expert_num = self.exp_per_task + self.shared_num

            linear_out = self.gate[keys[i]](input_data[i])
            cur_gate = F.softmax(linear_out, dim=-1)
            cur_gate = cur_gate.reshape(-1, cur_expert_num, 1)

            # f^{k}(x) = sum_{i=1}^{n}(g^{k}(x)_{i} * f_{i}(x))
            cur_experts = expert_outputs[i * self.exp_per_task:(i + 1) * self.exp_per_task] + \
                          expert_outputs[-int(self.shared_num):]

            expert_concat = torch.cat(cur_experts, 1)
            expert_concat = expert_concat.reshape(
                -1, cur_expert_num, self.expert_size)

            cur_gate_expert = expert_concat * cur_gate
            cur_gate_expert = cur_gate_expert.sum(1)
            outputs.append(cur_gate_expert)

        # shared gate part
        if not self.if_last:
            cur_expert_num = self.task_num * self.exp_per_task + self.shared_num

            linear_out = self.gate_shared(input_data[-1])
            cur_gate = F.softmax(linear_out, dim=-1)
            cur_gate = cur_gate.reshape(-1, cur_expert_num, 1)

            cur_experts = expert_outputs
            expert_concat = torch.cat(cur_experts, 1)
            expert_concat = expert_concat.reshape(
                -1, cur_expert_num, self.expert_size)

            cur_gate_expert = expert_concat * cur_gate
            cur_gate_expert = cur_gate_expert.sum(1)
            outputs.append(cur_gate_expert)

        return outputs
