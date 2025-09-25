import torch
import string
import numpy as np
from collections import defaultdict

class TensorNode:
    def __init__(self, tensor_or_shape, dim_labels, l=None, r=None, name=None, dtype=None):
        """Initializes a TensorNode object with the given shape and dimension labels."""
        if isinstance(tensor_or_shape, tuple) or isinstance(tensor_or_shape, list):
            self.tensor = torch.randn(tensor_or_shape, dtype=dtype)
            self.tensor = self.tensor / torch.norm(self.tensor)
        else:
            self.tensor = tensor_or_shape
        self.dim_labels = list(dim_labels)
        self.left_labels = [l] if isinstance(l, str) else (l or [])
        self.right_labels = [r] if isinstance(r, str) else (r or [])
        self.name = name or ''
        self.connections = {}
        self.connection_priority = defaultdict(float)
        self.contracted = set()

    def reset_connections(self):
        """Resets the connections of the node."""
        self.connections = {}
        self.connection_priority = defaultdict(float)
        self.contracted = set()

    def contract_with(self, other_node, contract_labels=None):
        """Contracts self with other_node over given dimensions, transferring priorities."""
        if self is other_node:
            return self
        if contract_labels is None:
            contract_labels = self.get_connecting_labels(other_node)
        contract_labels = [contract_labels] if isinstance(contract_labels, str) else contract_labels

        # Generate Einstein summation notation
        all_labels = self.dim_labels + other_node.dim_labels
        unique_labels = set(all_labels)
        label_to_index = {label: string.ascii_letters[i] for i, label in enumerate(unique_labels)}

        einsum_self = ''.join(label_to_index[label] for label in self.dim_labels)
        einsum_other = ''.join(label_to_index[label] for label in other_node.dim_labels)
        einsum_output = ''.join(label_to_index[label] for label in unique_labels if label not in contract_labels)

        contracted_tensor = torch.einsum(f"{einsum_self},{einsum_other}->{einsum_output}", self.tensor, other_node.tensor)
        new_dim_labels = [label for label in unique_labels if label not in contract_labels]

        left_labels = [label for label in self.left_labels + other_node.left_labels if label not in contract_labels]
        right_labels = [label for label in self.right_labels + other_node.right_labels if label not in contract_labels]

        node = TensorNode(contracted_tensor, new_dim_labels, l=left_labels, r=right_labels, name=f"<{self.name}-{','.join(contract_labels)}-{other_node.name}>")
        node.contracted = self.contracted | other_node.contracted
        if not self.contracted:
            node.contracted.add(self)
        if not other_node.contracted:
            node.contracted.add(other_node)

        # Set connections in the new node, transferring priorities
        for label, connected_node in self.connections.items():
            if connected_node not in node.contracted:
                if label in node.connections:
                    node.connection_priority[label] = max(node.connection_priority[label], self.connection_priority[label])
                else:
                    node.connection_priority[label] = self.connection_priority[label]
                node.connections[label] = connected_node

        for label, connected_node in other_node.connections.items():
            if connected_node not in node.contracted:
                if label in node.connections:
                    node.connection_priority[label] = max(node.connection_priority[label], other_node.connection_priority[label])
                else:
                    node.connection_priority[label] = other_node.connection_priority[label]
                node.connections[label] = connected_node
        return node

    def cuda(self):
        """Moves the tensor to the GPU."""
        self.tensor = self.tensor.cuda()
        return self

    def to(self, device=None, dtype=None):
        """Moves the tensor to the given device and dtype."""
        self.tensor = self.tensor.to(device=device, dtype=dtype)
        return self

    def connect(self, other_node, labels, priority=float("-inf")):
        """Connects this node to another node with the given labels and priority."""
        labels = [labels] if isinstance(labels, str) else list(labels)
        for label in labels:
            if label in self.connections:
                # Update priority if the new priority is higher
                self.connection_priority[label] = max(self.connection_priority[label], priority)
            else:
                self.connection_priority[label] = priority
            self.connections[label] = other_node

            if label in other_node.connections:
                other_node.connection_priority[label] = max(other_node.connection_priority[label], priority)
            else:
                other_node.connection_priority[label] = priority
            other_node.connections[label] = self

    def get_connecting_labels(self, other_node, horizontal=True):
        """Returns the labels that connect this node to another node."""
        if not self.contracted:
            return list(set([label for label, node in self.connections.items() if node == other_node and (horizontal or label not in self.left_labels + self.right_labels)] + [label for label, node in other_node.connections.items() if node == self and (horizontal or label not in other_node.left_labels + other_node.right_labels)]))
        # else, go through each node in the contracted set and accumulate the labels to which there is a connection
        connecting_labels = set()
        for node in self.contracted:
            for node2 in other_node.contracted | {other_node}:
                connecting_labels.update(node.get_connecting_labels(node2, horizontal))

        return list(connecting_labels)

    @property
    def shape(self):
        """Returns the shape of the tensor."""
        return self.tensor.shape

    def dim_size(self, label):
        """Returns the size of the dimension corresponding to the given label."""
        return self.tensor.shape[self.dim_labels.index(label)]
    
    def is_horizontal_bond(self, label):
        """Checks if the given label is a horizontal bond."""
        return label in self.left_labels or label in self.right_labels
    
    def sum_labels(self, labels):
        """Sums the tensor over the given labels."""
        if isinstance(labels, str):
            labels = [labels]
        indices = [self.dim_labels.index(label) for label in labels if label in self.dim_labels]
        return self.tensor.sum(indices)

    def squeeze(self, exclude=set()):
        """Squeezes the tensor and removes singleton dimensions."""
        # Remove singleton dimensions
        singleton = [s <= 1 and l not in exclude and l not in self.connections for s, l in zip(self.shape, self.dim_labels)]
        if any(singleton):
            squeezed_labels = [label for label, s in zip(self.dim_labels, singleton) if s]
            squeezed_indices = [i for i, s in enumerate(singleton) if s]
            self.dim_labels = [label for label, s in zip(self.dim_labels, singleton) if not s]
            self.tensor = self.tensor.squeeze(*squeezed_indices)
            # Remove sqeezed labels from left and right labels
            self.left_labels = [label for label in self.left_labels if label not in squeezed_labels]
            self.right_labels = [label for label in self.right_labels if label not in squeezed_labels]
        return self

    def contract_vertically(self, exclude=set()):
        """Contracts all non-left/right connections iteratively, prioritizing high-priority connections."""
        contracted = self
        contraction_queue = [self]

        while contraction_queue:
            current = contraction_queue.pop(0)
            # Sort connections by priority (descending)
            sorted_connections = sorted(current.connections.items(), key=lambda item: current.connection_priority[item[0]], reverse=True)

            for label, next_node in sorted_connections:
                if next_node in exclude:
                    continue

                if label not in current.left_labels + current.right_labels:
                    contracted = current.contract_with(next_node, next_node.get_connecting_labels(current, horizontal=False))
                    contraction_queue.append(contracted)

        return contracted

    def get_transposed_node(self, exclude=set()):
        """Returns the same node but with dummy dimension labels"""
        return TensorNode(self.tensor, [f'_{l}' if l not in exclude else l for l in self.dim_labels], l=[f'_{l}' if l not in exclude else l for l in self.left_labels], r=[f'_{l}' if l not in exclude else l for l in self.right_labels], name='_'+self.name)

    def copy(self):
        """Returns a copy of the node. Doesn't copy connections."""
        node = TensorNode(self.tensor, self.dim_labels.copy(), l=self.left_labels.copy(), r=self.right_labels.copy(), name=self.name+'_c')
        return node

    def update_node(self, step, lr=1.0, adaptive_step=False, min_norm=None, max_norm=None):
        """Updates the tensor of the node with the given step size."""
        if adaptive_step:
            # Compute update norm for adaptive step size
            step_norm = torch.norm(step)
            param_norm = torch.norm(self.tensor)
            
            if step_norm > param_norm:
                # Scale down large updates
                step = step * (param_norm / step_norm)
        
        # Apply update with learning rate
        new_tensor = self.tensor + lr * step
        # if min_norm is not None:
        #     if torch.norm(step) < min_norm:
        #         return self
        if max_norm is not None:
            # Project updates to maintain numerical stability
            current_norm = torch.norm(new_tensor)
            if current_norm > max_norm:
                new_tensor = new_tensor * (max_norm / current_norm)


        #print('step:', torch.norm(step), 'new:', torch.norm(new_tensor), 'old:', torch.norm(self.tensor))
        self.tensor = new_tensor
        return self

    def set_tensor(self, tensor):
        """Sets the tensor of the node to the given tensor."""
        self.tensor = tensor
        return self

    def permute_first(self, *labels, expand=True):
        """Permutes the tensor so that the given labels are first."""
        new_labels = list(labels) + [l for l in self.dim_labels if l not in labels]
        if not expand:
            new_labels = [l for l in new_labels if l in self.dim_labels]
        permute = [self.dim_labels.index(l) for l in new_labels if l in self.dim_labels]
        if permute:
            self.tensor = self.tensor.permute(*permute)
        if expand:
            for l in labels:
                if l not in self.dim_labels:
                    self.tensor = self.tensor.unsqueeze(new_labels.index(l))
        self.dim_labels = new_labels
        return self

    def permute_last(self, *labels):
        """Permutes the tensor so that the given labels are last."""
        new_labels = [l for l in self.dim_labels if l not in labels] + list(labels)
        permute = [self.dim_labels.index(l) for l in new_labels if l in self.dim_labels]
        self.tensor = self.tensor.permute(*permute)
        for l in labels:
            if l not in self.dim_labels:
                self.tensor = self.tensor.unsqueeze(new_labels.index(l))
        self.dim_labels = new_labels
        return self

    def permute(self, *labels):
        """Permutes the tensor according to the given labels."""
        permute = [self.dim_labels.index(l) for l in labels]
        self.tensor = self.tensor.permute(*permute)
        self.dim_labels = list(labels)
        return self

    def expand_labels(self, labels, size):
        """Expands the labels up to the given size only for the given labels"""
        # If label not in dim_labels, add a new dimension
        for label in labels:
            if label not in self.dim_labels:
                self.tensor = self.tensor.unsqueeze(-1)
                self.dim_labels = self.dim_labels + [label]
        # Expand the dimension to the given size
        sizes = [size[labels.index(l)] if l in labels else -1 for l in self.dim_labels]
        self.tensor = self.tensor.expand(*sizes)
        return self

    def get_vertical_nodes(self, exclude=set()):
        """Returns the nodes in the order they would be contracted vertically, without contracting."""
        order = []
        visited = set()
        queue = [self]

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            order.append(current)
            # Sort connections by priority (descending)
            sorted_connections = sorted(
                current.connections.items(),
                key=lambda item: current.connection_priority[item[0]],
                reverse=True
            )
            for label, next_node in sorted_connections:
                if next_node in exclude or next_node in visited:
                    continue
                if label not in current.left_labels + current.right_labels:
                    queue.append(next_node)
        return order

    def __repr__(self):
        """String representation of the TensorNode."""
        return f"TensorNode(name={self.name}, shape={self.shape}, labels={self.dim_labels})"
    
class CPDTensorNode(TensorNode):
    def update_node(self, step, lr=1):
        """Updates the tensor of the node with the given step size."""
        if not self.left_labels or not self.right_labels:
            self.tensor = self.tensor + lr * step
        else:
            left = self.dim_size(self.left_labels[0])
            mask = torch.eye(left, device=self.tensor.device, dtype=self.tensor.dtype)
            for l in self.dim_labels:
                if self.is_horizontal_bond(l):
                    continue
                mask = mask.unsqueeze(self.dim_labels.index(l))
            mask = mask.expand_as(step)
            self.tensor = self.tensor + lr * (step * mask)
        return self
