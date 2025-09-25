import torch
import string
from torch import nn
import string
from collections import deque
from tensor.node import TensorNode
from tqdm.auto import tqdm
import time
import numpy as np
from scipy.sparse.linalg import LinearOperator, cg
from tensor.utils import EinsumLabeler

class TensorNetwork:
    def __init__(self, input_nodes, main_nodes, train_nodes=None, output_labels=('s',), sample_dim='s'):
        """Initializes a TensorNetwork with ordered input and main nodes."""
        self.input_nodes = input_nodes
        self.main_nodes = main_nodes
        self.train_nodes = main_nodes if train_nodes is None else train_nodes
        #self.node_indices = {node: i for i, node in enumerate(main_nodes)}
        self.left_stacks = None
        self.right_stacks = None
        self.output_labels = output_labels
        self.sample_dim = sample_dim
        self.nodes, self.node_indices = self._discover_nodes()

    def cuda(self):
        """Moves all tensors to the GPU."""
        for node in self.nodes:
            node.cuda()
        return self

    def to(self, device=None, dtype=None):
        """Moves all tensors to the given device and dtype."""
        for node in self.nodes:
            node.to(device=device, dtype=dtype)
        return self

    def _discover_nodes(self):
        """Uses BFS to find all connected nodes from input and main nodes."""
        node_indices = {node: i for i, node in enumerate(self.main_nodes)}
        discovered = set(self.main_nodes)
        queue = deque(self.main_nodes)

        while queue:
            node = queue.popleft()
            current_index = node_indices[node]
            for label, connected_node in node.connections.items():
                if connected_node not in discovered and not node.is_horizontal_bond(label):
                    discovered.add(connected_node)
                    queue.append(connected_node)
                    node_indices[connected_node] = current_index

        return list(sorted(discovered, key=lambda n: n.name)), node_indices

    def compute_stacks(self, direction="left", exclude_nodes=set()):
        """Computes left or right stacks iteratively."""
        stack_dict = {}
        nodes = self.main_nodes if direction == "left" else reversed(self.main_nodes)

        prev_stack = None

        for node in nodes:
            column_nodes = [node] + self.get_column_nodes(node)
            node_iter = iter(column_nodes)
            contracted = next(node_iter) if prev_stack is None else prev_stack
            for vnode in node_iter:
                contracted = contracted.contract_with(vnode, vnode.get_connecting_labels(contracted))
            stack_dict[node] = contracted
            prev_stack = contracted

        return stack_dict

    def recompute_all_stacks(self, exclude_nodes=set()):
        """Computes left and right stacks for each node."""
        self.left_stacks = self.compute_stacks("left", exclude_nodes)
        self.right_stacks = self.compute_stacks("right", exclude_nodes)

    def reset_stacks(self, node=None):
        """Resets the left and right stacks to None."""
        self.left_stacks = None
        self.right_stacks = None

    def get_stacks(self, node):
        """Returns the left and right stacks for the given node."""
        index = self.node_indices[node]
        left_stack = self.left_stacks[self.main_nodes[index - 1]] if index > 0 else None
        right_stack = self.right_stacks[self.main_nodes[index + 1]] if index < len(self.main_nodes) - 1 else None
        return left_stack, right_stack

    def get_column_nodes(self, node):
        """Returns the column nodes for the given index."""
        column_nodes = []
        index = self.node_indices[node]
        for n, i in self.node_indices.items():
            if n is node:
                continue
            if i == index:
                column_nodes.append(n)
        return column_nodes

    def compute_jacobian_stack(self, node) -> TensorNode:
        """Computes the jacobian of a node by contracting all non-left/right connections separately."""

        left_stack, right_stack = self.get_stacks(node)
        column_nodes = self.get_column_nodes(node)
        node_iter = iter(column_nodes)
        contracted = next(node_iter) if left_stack is None else left_stack
        for vnode in node_iter:
            contracted = contracted.contract_with(vnode, vnode.get_connecting_labels(contracted))
        if right_stack is not None:
            contracted = contracted.contract_with(right_stack, right_stack.get_connecting_labels(contracted))

        return contracted

    def forward(self, x, to_tensor=False):
        """Computes the forward pass of the network."""
        self.set_input(x)

        if self.left_stacks is None or self.right_stacks is None:
            self.recompute_all_stacks()
        node = self.main_nodes[0]
        column_nodes = [node]+self.get_column_nodes(node)
        left_stack, right_stack = self.get_stacks(node)

        node_iter = iter(column_nodes)
        contracted = next(node_iter) if left_stack is None else left_stack

        for vnode in node_iter:
            contracted = contracted.contract_with(vnode, vnode.get_connecting_labels(contracted))
        if right_stack is not None:
            contracted = contracted.contract_with(right_stack, right_stack.get_connecting_labels(contracted))

        if self.output_labels is not None:
            contracted = contracted.permute_first(*self.output_labels)
        if to_tensor:
            contracted = contracted.tensor
        return contracted

    def forward_batch(self, x, batch_size):
        """Computes the forward pass of the network in batches."""
        data_size = len(x) if isinstance(x, torch.Tensor) else x[0].shape[0]
        if batch_size <= 0 or batch_size >= data_size:
            return self.forward(x, to_tensor=True)
        batches = (data_size + batch_size - 1) // batch_size # round up division
        outputs = []
        for b in range(batches):
            x_batch = x[b*batch_size:(b+1)*batch_size] if isinstance(x, torch.Tensor) else [x[i][b*batch_size:(b+1)*batch_size] for i in range(len(x))]
            y_batch = self.forward(x_batch, to_tensor=True)
            outputs.append(y_batch)
        return torch.cat(outputs, dim=0)

    def left_update_stacks(self, node):
        """Updates the left stacks."""
        previous_stack, _ = self.get_stacks(node)
        column_nodes = [node]+self.get_column_nodes(node)
        node_iter = iter(column_nodes)
        contracted = next(node_iter) if previous_stack is None else previous_stack
        for vnode in node_iter:
            contracted = contracted.contract_with(vnode, vnode.get_connecting_labels(contracted))

        self.left_stacks[node] = contracted

    def right_update_stacks(self, node):
        """Updates the right stacks."""
        _, next_stack = self.get_stacks(node)
        column_nodes = [node]+self.get_column_nodes(node)
        node_iter = iter(reversed(column_nodes))
        contracted = next(node_iter) if next_stack is None else next_stack
        for vnode in node_iter:
            contracted = contracted.contract_with(vnode, vnode.get_connecting_labels(contracted))

        self.right_stacks[node] = contracted

    @torch.no_grad()
    def get_A_b(self, node, grad, hessian, method=None):
        """Finds the update step for a given node"""

        # Determine broadcast
        broadcast_dims = tuple(d for d in self.output_labels if d not in node.dim_labels)
        non_broadcast_dims = tuple(d for d in self.output_labels if d != self.sample_dim)

        # Compute the Jacobian
        J = self.compute_jacobian_stack(node).copy().expand_labels(self.output_labels, grad.shape).permute_first(*broadcast_dims)

        # Assign unique einsum labels
        dim_labels = EinsumLabeler()

        dd_loss_ein = ''.join([dim_labels[self.sample_dim]] + [dim_labels[d] for d in non_broadcast_dims] + [dim_labels['_' + d] for d in non_broadcast_dims])
        d_loss_ein = ''.join(dim_labels[d] for d in self.output_labels)

        J_ein1 = ''
        J_ein2 = ''
        J_out1 = []
        J_out2 = []
        dim_order = []
        for d in J.dim_labels:
            J_ein1 += dim_labels[d]
            J_ein2 += dim_labels['_' + d] if d != self.sample_dim else dim_labels[d]
            if d not in broadcast_dims:
                J_out1.append(dim_labels[d])
                J_out2.append(dim_labels['_' + d])
                dim_order.append(d)
        J_out1 = ''.join([J_out1[dim_order.index(d)] for d in node.dim_labels])
        J_out2 = ''.join([J_out2[dim_order.index(d)] for d in node.dim_labels])

        # Construct einsum notations
        einsum_A = f'{J_ein1},{J_ein2},{dd_loss_ein}->{J_out1}{J_out2}'
        einsum_b = f"{J_ein1},{d_loss_ein}->{J_out1}"

        # Compute einsum operations
        if method is None:
            A = torch.einsum(einsum_A, J.tensor.conj(), J.tensor, hessian)
        else:
            A = torch.randn((2,2,2,2))
        b = torch.einsum(einsum_b, J.tensor.conj(), grad)

        return A, b

    def get_J(self, node, grad):
        # Determine broadcast
        broadcast_dims = tuple(d for d in self.output_labels if d not in node.dim_labels)
        non_broadcast_dims = tuple(d for d in self.output_labels if d != self.sample_dim)

        # Compute the Jacobian
        J = self.compute_jacobian_stack(node).copy().expand_labels(self.output_labels, grad.shape).permute_first(*broadcast_dims)

        # Assign unique einsum labels
        all_letters = iter(string.ascii_letters)
        dim_labels = {dim: next(all_letters) for dim in self.output_labels}  # Assign letters to output dims
        for d in non_broadcast_dims:
            dim_labels['_' + d] = next(all_letters)

        d_loss_ein = ''.join(dim_labels[d] for d in self.output_labels)
        dd_loss_ein = ''.join([dim_labels[self.sample_dim]] + [dim_labels[d] for d in non_broadcast_dims] + [dim_labels['_' + d] for d in non_broadcast_dims])
        coeff_ein = ''.join([dim_labels[self.sample_dim]] + [dim_labels['_' + d] for d in non_broadcast_dims])
        J_ein1 = ''
        J_out1 = []
        dim_order = []
        for d in J.dim_labels:
            if d not in dim_labels and d not in broadcast_dims:
                dim_labels[d] = next(all_letters)
                dim_labels['_' + d] = next(all_letters)
            J_ein1 += dim_labels[d]
            if d not in broadcast_dims:
                J_out1.append(dim_labels[d])
                dim_order.append(d)
        J_out1 = ''.join([J_out1[dim_order.index(d)] for d in node.dim_labels])

        return {
            'J': J.permute_first(dim_order, expand=False),
            'einsum': J_ein1,
            'node_ein': J_out1,
            'dd_loss_ein': dd_loss_ein,
            'd_loss_ein': d_loss_ein,
            'coeff_ein': coeff_ein,
        }

    def get_b(self, node, grad):
        # Determine broadcast
        broadcast_dims = tuple(d for d in self.output_labels if d not in node.dim_labels)
        non_broadcast_dims = tuple(d for d in self.output_labels if d != self.sample_dim)

        # Compute the Jacobian
        J = self.compute_jacobian_stack(node).copy().expand_labels(self.output_labels, grad.shape).permute_first(*broadcast_dims)

        # Assign unique einsum labels
        import string
        all_letters = iter(string.ascii_letters)
        dim_labels = {dim: next(all_letters) for dim in self.output_labels}  # Assign letters to output dims
        for d in non_broadcast_dims:
            dim_labels['_' + d] = next(all_letters)

        d_loss_ein = ''.join(dim_labels[d] for d in self.output_labels)
        J_ein1 = ''
        J_out1 = []
        dim_order = []
        for d in J.dim_labels:
            if d not in dim_labels and d not in broadcast_dims:
                dim_labels[d] = next(all_letters)
                dim_labels['_' + d] = next(all_letters)
            J_ein1 += dim_labels[d]
            if d not in broadcast_dims:
                J_out1.append(dim_labels[d])
                dim_order.append(d)
        J_out1 = ''.join([J_out1[dim_order.index(d)] for d in node.dim_labels])
        einsum_b = f"{J_ein1},{d_loss_ein}->{J_out1}"

        # Compute einsum operations
        b = torch.einsum(einsum_b, J.tensor, grad)

        return b

    def solve_system(self, node, A, b, method='exact', eps=0.0):
        """Finds the update step for a given node"""
        # Solve the system
        A_f = A.flatten(0, A.ndim//2-1).flatten(1, -1)
        b_f = b.flatten()
        scale = A_f.diag().abs().mean()
        if scale == 0:
            scale = 1
        A_f = A_f / scale
        b_f = b_f / scale

        if method.lower() == 'exact':
            x = torch.linalg.solve(A_f, -b_f)
        elif method.lower() == 'ridge_exact':
            ##A_f.diagonal(dim1=-2, dim2=-1).add_(2 * eps)
            A_f = A_f + (2 * eps) * torch.eye(A_f.shape[-1], dtype=A_f.dtype, device=A_f.device)
            b_f = b_f + (2 * eps) * node.tensor.flatten()
            x = torch.linalg.solve(A_f, -b_f)
        elif method.lower().startswith('ridge_cholesky'):
            A_f = A_f + (2 * eps) * torch.eye(A_f.shape[-1], dtype=A_f.dtype, device=A_f.device)
            b_f = b_f + (2 * eps) * node.tensor.flatten()
            L = torch.linalg.cholesky(A_f)
            x = torch.cholesky_solve(-b_f.unsqueeze(-1), L)
            x = x.squeeze(-1)
        elif method.lower() == 'cholesky':
            L = torch.linalg.cholesky(A_f)
            x = torch.cholesky_solve(-b_f.unsqueeze(-1), L)
            x = x.squeeze(-1)
        elif method.lower() == 'gradient':
            x = -b
        else:
            raise ValueError(f"Unknown method: {method}")

        step_tensor = x.reshape(b.shape)
        return step_tensor

    def set_input(self, x):
        """Sets the input tensor for the network."""
        was_updated = False
        if isinstance(x, tuple) or isinstance(x, list):
            for node, tensor in zip(self.input_nodes, x):
                if node.tensor is not tensor:
                    was_updated = True
                    node.set_tensor(tensor)
        else:
            for node in self.input_nodes:
                if node.tensor is not x:
                    was_updated = True
                    node.set_tensor(x)
        if was_updated:
            self.left_stacks = None
            self.right_stacks = None
        return was_updated

    def disconnect(self, nodes):
        """Creates a new TensorNetwork without the given nodes and their connections.
        The nodes themselves are virtual copies with their own connections dictionary.
        Only the underlying tensors are carried over."""
        if not isinstance(nodes, list) and not isinstance(nodes, tuple):
            nodes = [nodes]

        # Create a mapping of old nodes to new virtual copies
        node_mapping = {}
        for node in self.nodes:
            if node not in nodes:
                # Create a virtual copy of the node with the same tensor
                new_node = TensorNode(node.tensor, node.dim_labels, l=node.left_labels, r=node.right_labels, name=node.name)
                node_mapping[node] = new_node

        # Recreate connections in the new virtual nodes
        for old_node, new_node in node_mapping.items():
            for label, connected_node in old_node.connections.items():
                if connected_node in node_mapping:  # Only add connections to nodes that are not being removed
                    new_node.connections[label] = node_mapping[connected_node]
                    new_node.connection_priority[label] = old_node.connection_priority[label]

        # Create new input and main nodes lists
        new_input_nodes = [node_mapping[node] for node in self.input_nodes if node in node_mapping]
        new_main_nodes = [node_mapping[node] for node in self.main_nodes if node in node_mapping]
        new_train_nodes = [node_mapping[node] for node in self.train_nodes if node in node_mapping]

        # Create a new TensorNetwork object
        new_network = TensorNetwork(new_input_nodes, new_main_nodes, new_train_nodes, self.output_labels, self.sample_dim)

        return new_network

    def accumulating_swipe(self, x, y_true, loss_fn, node_order=None, batch_size=-1, num_swipes=1, lr=1.0, method='exact', eps=1e-12, eps_decay=None, convergence_criterion=None, orthonormalize=False, verbose=False, skip_second=False, blocks_input=False, timeout=None, data_device=None, model_device=None, disable_tqdm=None, block_callback=None, loss_callback=None, direction='l2r', update_or_reset_stack='reset', adaptive_step=False, min_norm=None, max_norm=None, eps_per_node=False):
        """Swipes the network to minimize the loss using accumulated A and b over mini-batches.
        Args:
            timeout (float or None): Maximum time in seconds to run. If None, no timeout.
            data_device (torch.device or None): Device where the data is stored (CPU or GPU).
            model_device (torch.device or None): Device where the model is (typically GPU).
        """
        data_size = len(x) if isinstance(x, torch.Tensor) else x[0].shape[0]
        if batch_size <= 0:
            batch_size = data_size
        batches = (data_size + batch_size - 1) // batch_size # round up division
        if blocks_input:
            batches = 1

        node_l2r = None
        node_r2l = None
        start_time = time.time() if timeout is not None else None

        def move_batch(batch):
            if model_device is not None and data_device is not None and data_device != model_device:
                if isinstance(batch, torch.Tensor) and batch.device != model_device:
                    return batch.to(model_device, non_blocking=True)
                elif isinstance(batch, (list, tuple)):
                    return [b.to(model_device, non_blocking=True) if b.device != model_device else b for b in batch]
            return batch

        # If disable_tqdm is not set, default to verbose < 2
        if disable_tqdm is None:
            disable_tqdm = verbose < 2

        NS = 0
        for _ in (tbar:=tqdm(range(num_swipes), disable=disable_tqdm)):
            if isinstance(eps, list):
                eps_ = eps[NS]
            else:
                eps_ = eps
            if eps_decay is not None:
                eps_ = eps_ * eps_decay**NS
            # LEFT TO RIGHT
            if node_order is not None:
                if isinstance(node_order, tuple):
                    first_node_order = node_order[0]
                else:
                    first_node_order = node_order
            else:
                first_node_order = self.train_nodes
            first_node_order = list(first_node_order if direction == 'l2r' else reversed(first_node_order))
            for node_i, node_l2r in enumerate(first_node_order):
                if eps_per_node:
                    if isinstance(eps, list):
                        eps_ = eps[node_i if direction == 'l2r' else len(first_node_order)-1-node_i]
                    else:
                        eps_ = eps
                if node_l2r in self.node_indices and node_r2l in self.node_indices and self.node_indices[node_l2r] == self.node_indices[node_r2l]:
                    continue
                # Timeout check
                if timeout is not None and (time.time() - start_time) > timeout:
                    print(f"Timeout reached ({timeout} seconds). Stopping accumulating_swipe.")
                    return False
                A_out, b_out = None, None
                total_loss = 0.0

                for b in tqdm(range(batches), desc=f"Left to right pass ({node_l2r.name if hasattr(node_l2r, 'name') else 'node'})", disable=disable_tqdm):
                    # Timeout check inside batch loop
                    if timeout is not None and (time.time() - start_time) > timeout:
                        print(f"Timeout reached ({timeout} seconds). Stopping accumulating_swipe.")
                        return False
                    if blocks_input or batch_size == data_size:
                        x_batch = x
                        y_batch = y_true
                    else:
                        x_batch = x[b*batch_size:(b+1)*batch_size] if isinstance(x, torch.Tensor) else [x[i][b*batch_size:(b+1)*batch_size] for i in range(len(x))]
                        y_batch = y_true[b*batch_size:(b+1)*batch_size]

                    x_batch = move_batch(x_batch)
                    y_batch = move_batch(y_batch)

                    y_pred = self.forward(x_batch, to_tensor=True)
                    loss, d_loss, sqd_loss = loss_fn.forward(y_pred, y_batch)
                    if method == 'gradient':
                        A, b_vec = self.get_A_b(node_l2r, d_loss, sqd_loss, method=method)
                    else:
                        A, b_vec = self.get_A_b(node_l2r, d_loss, sqd_loss)

                    if A_out is None:
                        A_out = A
                        b_out = b_vec
                    else:
                        A_out.add_(A)
                        b_out.add_(b_vec)
                    if method == 'gradient':
                        node_l2r.update_node(b_vec, lr=lr, adaptive_step=adaptive_step, min_norm=min_norm, max_norm=max_norm)

                    total_loss += loss.mean().item()
                if verbose > 1:
                    print(f"NS: {NS}, Left loss ({node_l2r.name if hasattr(node_l2r, 'name') else 'node'}):", total_loss / batches, f" (eps: {eps_})")
                try:
                    # Choose exact if eps is 0
                    _method = method
                    if eps_ == 0 and method == 'ridge_exact':
                        _method = 'exact'
                    step_tensor = self.solve_system(node_l2r, A_out, b_out, method=_method, eps=eps_)
                except torch.linalg.LinAlgError:
                    if verbose > 0:
                        print(f"Singular system for node {node_l2r.name if hasattr(node_l2r, 'name') else 'node'}")
                    return False
                if method != 'gradient':
                    node_l2r.update_node(step_tensor, lr=lr, adaptive_step=adaptive_step, min_norm=min_norm, max_norm=max_norm)
                if orthonormalize:
                    self.node_orthonormalize_left(node_l2r)
                if update_or_reset_stack == 'reset':
                    self.reset_stacks(node_l2r)
                elif update_or_reset_stack == 'update':
                    self.left_update_stacks(node_l2r)
                if loss_callback is not None:
                    loss_callback(NS, node_l2r, total_loss / batches)

                # Convergence check after node update (pause timer here)
                if convergence_criterion is not None:
                    if convergence_criterion():
                        if verbose > 0:
                            print('Converged (left pass)')
                        if block_callback is not None:
                            block_callback(NS, node_l2r)
                        return True
                if block_callback is not None:
                    block_callback(NS, node_l2r)
            NS += 1
            if not disable_tqdm:
                tbar.set_postfix_str(f"NS: {NS}, eps: {eps_}")
            if skip_second:
                continue

            if isinstance(eps, list):
                eps_ = eps[NS]
            else:
                eps_ = eps
            if eps_decay is not None:
                eps_ = eps_ * eps_decay**NS

            # RIGHT TO LEFT
            if node_order is not None:
                if isinstance(node_order, tuple):
                    second_node_order = node_order[1]
                else:
                    second_node_order = reversed(node_order)
            else:
                second_node_order = self.train_nodes
            second_node_order = list(second_node_order if direction == 'r2l' else reversed(list(second_node_order)))
            for node_i, node_r2l in enumerate(second_node_order):
                if eps_per_node:
                    if isinstance(eps, list):
                        eps_ = eps[node_i if direction == 'r2l' else len(second_node_order)-1-node_i]
                    else:
                        eps_ = eps
                if node_l2r in self.node_indices and node_r2l in self.node_indices and self.node_indices[node_l2r] == self.node_indices[node_r2l]:
                    continue
                # Timeout check
                if timeout is not None and (time.time() - start_time) > timeout:
                    print(f"Timeout reached ({timeout} seconds). Stopping accumulating_swipe.")
                    return False
                A_out, b_out = None, None
                total_loss = 0.0

                for b in tqdm(range(batches), desc=f"Right to left pass ({node_r2l.name if hasattr(node_r2l, 'name') else 'node'})", disable=disable_tqdm):
                    # Timeout check inside batch loop
                    if timeout is not None and (time.time() - start_time) > timeout:
                        print(f"Timeout reached ({timeout} seconds). Stopping accumulating_swipe.")
                        return False
                    if blocks_input or batch_size == data_size:
                        x_batch = x
                        y_batch = y_true
                    else:
                        x_batch = x[b*batch_size:(b+1)*batch_size] if isinstance(x, torch.Tensor) else [x[i][b*batch_size:(b+1)*batch_size] for i in range(len(x))]
                        y_batch = y_true[b*batch_size:(b+1)*batch_size]

                    x_batch = move_batch(x_batch)
                    y_batch = move_batch(y_batch)

                    y_pred = self.forward(x_batch, to_tensor=True)
                    loss, d_loss, sqd_loss = loss_fn.forward(y_pred, y_batch)

                    A, b_vec = self.get_A_b(node_r2l, d_loss, sqd_loss)
                    if A_out is None:
                        A_out = A
                        b_out = b_vec
                    else:
                        A_out.add_(A)
                        b_out.add_(b_vec)

                    total_loss += loss.mean().item()

                if verbose > 1:
                    print(f"NS: {NS}, Right loss ({node_r2l.name if hasattr(node_r2l, 'name') else 'node'}):", total_loss / batches, f" (eps: {eps_})")
                try:
                    # Choose exact if eps is 0
                    _method = method
                    if eps_ == 0 and method == 'ridge_exact':
                        _method = 'exact'
                    step_tensor = self.solve_system(node_r2l, A_out, b_out, method=_method, eps=eps_)
                except torch.linalg.LinAlgError:
                    if verbose > 0:
                        print(f"Singular system for node {node_r2l.name if hasattr(node_r2l, 'name') else 'node'}")
                    return False

                node_r2l.update_node(step_tensor, lr=lr, adaptive_step=adaptive_step, min_norm=min_norm, max_norm=max_norm)
                if orthonormalize:
                    self.node_orthonormalize_right(node_r2l)
                if update_or_reset_stack == 'reset':
                    self.reset_stacks(node_r2l)
                elif update_or_reset_stack == 'update':
                    self.right_update_stacks(node_r2l)
                if loss_callback is not None:
                    loss_callback(NS, node_r2l, total_loss / batches)

                # Convergence check after node update (pause timer here)
                if convergence_criterion is not None:
                    if convergence_criterion():
                        if verbose > 0:
                            print('Converged (right pass)')
                        if block_callback is not None:
                            block_callback(NS, node_r2l)
                        return True
                if block_callback is not None:
                    block_callback(NS, node_r2l)
            NS += 1
            if not disable_tqdm:
                tbar.set_postfix_str(f"NS: {NS}, eps: {eps_}")

        return True


    def orthonormalize_left(self):
        """
        Orthonormalizes the left bonds of the main nodes using QR decomposition.
        """
        for n in self.main_nodes:
            self.node_orthonormalize_left(n)

    def orthonormalize_right(self):
        """
        Orthonormalizes the right bonds of the main nodes using QR decomposition.
        """
        for n in self.main_nodes:
            self.node_orthonormalize_right(n)

    def node_orthonormalize_left(self, node):
        index = self.node_indices[node]
        if index >= len(self.main_nodes) - 1:
            return

        # Get bond dimension indices
        right_bond_indices = [node.dim_labels.index(lbl) for lbl in node.right_labels]
        all_indices = list(range(len(node.shape)))
        left_nonbond_indices = [idx for idx in all_indices if idx not in right_bond_indices]

        # Permute so that left dims are first, right bond last
        perm_order = left_nonbond_indices + right_bond_indices
        A_permuted = node.tensor.permute(perm_order)

        # Flatten into a 2D matrix: (left combined, right bond)
        original_shape = A_permuted.shape
        A_reshaped = A_permuted.flatten(0, len(left_nonbond_indices)-1)

        # Perform QR decomposition
        Q, R = torch.linalg.qr(A_reshaped, mode='reduced')

        # Reshape Q back
        Q = Q.reshape(original_shape[:len(left_nonbond_indices)] + (Q.shape[-1],))

        # Reverse the permutation
        inverse_perm = sorted(range(len(perm_order)), key=lambda k: perm_order[k])
        node.tensor = Q.permute(*inverse_perm)

        # Push R into the next node's left bonds
        next_node = self.main_nodes[index + 1]
        bond_dims = node.get_connecting_labels(next_node)
        next_node.permute_first(*bond_dims)
        next_node.tensor = torch.einsum('ij,j...->i...', R, next_node.tensor)

        if self.right_stacks is not None:
            self.right_update_stacks(next_node)

    def node_orthonormalize_right(self, node):
        index = self.node_indices[node]
        if index <= 0:
            return

        # Identify left bond indices and the rest.
        left_bond_indices = [node.dim_labels.index(lbl) for lbl in node.left_labels]
        all_indices = list(range(len(node.shape)))
        right_nonbond_indices = [idx for idx in all_indices if idx not in left_bond_indices]

        # Permute so that left bonds come last (we flip the order to simulate RQ)
        # One strategy is to bring the nonbond dims first, then the bond dims.
        perm_order = right_nonbond_indices + left_bond_indices
        A_permuted = node.tensor.permute(perm_order)

        # Save original shape to later reshape Q back.
        original_shape = A_permuted.shape

        # Flatten: rows combine nonbond dims, columns combine left bond dims.
        A_reshaped = A_permuted.flatten(0, len(right_nonbond_indices)-1)

        # --- Simulate RQ decomposition ---
        # Reverse both dimensions.
        A_rev = torch.flip(A_reshaped, dims=[0, 1])
        Q_rev, R_rev = torch.linalg.qr(A_rev, mode='reduced')
        # Reverse factors back
        R = torch.flip(R_rev.T, dims=[0, 1])
        Q = torch.flip(Q_rev, dims=[0, 1])
        # --- End RQ ---

        # Reshape Q back to tensor form.
        Q = Q.reshape(original_shape[:len(right_nonbond_indices)] + (Q.shape[-1],))

        # Reverse the permutation
        inverse_perm = sorted(range(len(perm_order)), key=lambda k: perm_order[k])
        node.tensor = Q.permute(*inverse_perm)

        # Push R into the previous node's right bonds
        prev_node = self.main_nodes[index - 1]

        bond_dims = node.get_connecting_labels(prev_node)
        prev_node.permute_last(*bond_dims)
        prev_node.tensor = torch.einsum('ji,...j->...i', R, prev_node.tensor)

        if self.left_stacks is not None:
            self.left_update_stacks(prev_node)

    def lanczos_swipe(self, x, y_true, loss_fn, batch_size=1, num_swipes=1, lr=1.0, max_iter=50, tol=1e-6, verbose=False, timeout=None, data_device=None, model_device=None, disable_tqdm=None, block_callback=None, loss_callback=None):
        """
        Swipes the network to minimize the loss using the Lanczos algorithm for each node, without forming the full Gramian.
        Args:
            max_iter (int): Maximum Lanczos steps per node.
            tol (float): Residual tolerance for convergence.
            Other args as in accumulating_swipe.
        """
        import torch
        import time
        from tqdm.auto import tqdm

        data_size = len(x) if isinstance(x, torch.Tensor) else x[0].shape[0]
        if batch_size <= 0:
            batch_size = data_size
        batches = (data_size + batch_size - 1) // batch_size

        start_time = time.time() if timeout is not None else None

        def move_batch(batch):
            if model_device is not None and data_device is not None and data_device != model_device:
                if isinstance(batch, torch.Tensor):
                    return batch.to(model_device, non_blocking=True)
                elif isinstance(batch, (list, tuple)):
                    return [b.to(model_device, non_blocking=True) for b in batch]
            return batch

        if disable_tqdm is None:
            disable_tqdm = int(verbose) < 2

        for NS in range(num_swipes):
            for node in self.train_nodes if NS % 2 == 0 else reversed(self.train_nodes):
                if timeout is not None and (time.time() - start_time) > timeout:
                    print(f"Timeout reached ({timeout} seconds). Stopping lanczos_swipe.")
                    return False

                # Precompute batches of HJ and b for this node
                b_rhs = torch.zeros_like(node.tensor)
                d_losss = []
                dd_losss = []
                loss_total = 0.0
                for b in range(batches):
                    x_batch = x[b*batch_size:(b+1)*batch_size] if isinstance(x, torch.Tensor) else [x[i][b*batch_size:(b+1)*batch_size] for i in range(len(x))]
                    y_batch = y_true[b*batch_size:(b+1)*batch_size]
                    x_batch = move_batch(x_batch)
                    y_batch = move_batch(y_batch)

                    y_pred = self.forward(x_batch, to_tensor=True)
                    loss, d_loss, sqd_loss = loss_fn.forward(y_pred, y_batch)

                    b_vec = self.get_b(node, d_loss)
                    b_rhs.add_(b_vec)

                    d_losss.append(d_loss)
                    dd_losss.append(sqd_loss)
                    if loss_callback is not None:
                        loss_total += loss.mean().item()
                if loss_callback is not None:
                    loss_callback(loss_total / batches)

                # Helper: matvec for this node (A @ v)
                def matvec(v):
                    Av = 0
                    for b, d_loss, dd_loss in zip(range(batches), d_losss, dd_losss):
                        x_batch = x[b*batch_size:(b+1)*batch_size] if isinstance(x, torch.Tensor) else [x[i][b*batch_size:(b+1)*batch_size] for i in range(len(x))]
                        x_batch = move_batch(x_batch)


                        self.set_input(x_batch)
                        if self.left_stacks is None or self.right_stacks is None:
                           self.recompute_all_stacks()

                        prep_J = self.get_J(node, d_loss)
                        J = prep_J['J']
                        J_einsum = prep_J['einsum']
                        node_ein = prep_J['node_ein']
                        d_loss_ein = prep_J['d_loss_ein']
                        dd_loss_ein = prep_J['dd_loss_ein']
                        coeff_ein = prep_J['coeff_ein']
                        coeff = torch.einsum(f"{J_einsum},{node_ein},{dd_loss_ein}->{coeff_ein}", J.tensor, v, dd_loss)
                        Av += torch.einsum(f"{J_einsum},{d_loss_ein}->{node_ein}", J.tensor, coeff)
                    return Av

                # Initial guess x0 = zeros
                x0 = torch.randn_like(b_rhs)

                # Lanczos-Galerkin solver (minimal, 1D case)
                def lanczos_solver(matvec, b, x0, max_iter, tol):
                    v = [torch.zeros_like(x0)]
                    a = [0.0]
                    b_coeffs = [0.0]
                    r0 = b - matvec(x0)
                    beta1 = torch.norm(r0)
                    b_coeffs.append(beta1)
                    v1 = r0 / beta1
                    v.append(v1)
                    for j in tqdm(range(1, max_iter+1)):
                        w = matvec(v[j]) - b_coeffs[j] * v[j-1]
                        a_j = (w*v[j]).sum()
                        a.append(a_j)
                        w = w - a_j * v[j]
                        beta_j1 = torch.norm(w)
                        b_coeffs.append(beta_j1)
                        v.append(w / beta_j1)
                        if beta_j1 < tol:
                           break
                    Vm = torch.stack(v[1:j+1], dim=-1)  # (n, m)
                    Tm = torch.diag(torch.tensor(a[1:], device=x0.device, dtype=x0.dtype))
                    if len(a) > 2:
                        Tm += torch.diag(torch.tensor(b_coeffs[2:j+1], device=x0.device, dtype=x0.dtype), diagonal=1)
                        Tm += torch.diag(torch.tensor(b_coeffs[2:j+1], device=x0.device, dtype=x0.dtype), diagonal=-1)
                    rhs = torch.zeros(len(a)-1, device=x0.device, dtype=x0.dtype)
                    rhs[0] = beta1
                    y = torch.linalg.solve(Tm, rhs)
                    x = x0 + Vm @ y
                    return x

                # Solve for update step
                step_tensor = lanczos_solver(matvec, -b_rhs, x0, max_iter, tol)
                node.update_node(step_tensor, lr=lr)
                self.left_update_stacks(node)
                if block_callback is not None:
                    block_callback(NS, node)
        return True

    def scipy_swipe(self, x, y_true, loss_fn, solver, batch_size=1, num_swipes=1, lr=1.0, max_iter=50, tol=1e-6, verbose=False, timeout=None, data_device=None, model_device=None, disable_tqdm=None, block_callback=None, loss_callback=None):
        """
        Swipes the network to minimize the loss using SciPy for each node, without forming the full Gramian.
        Args:
            max_iter (int): Maximum Lanczos steps per node.
            tol (float): Residual tolerance for convergence.
            Other args as in accumulating_swipe.
        """

        data_size = len(x) if isinstance(x, torch.Tensor) else x[0].shape[0]
        if batch_size <= 0:
            batch_size = data_size
        batches = (data_size + batch_size - 1) // batch_size

        start_time = time.time() if timeout is not None else None

        def move_batch(batch):
            if model_device is not None and data_device is not None and data_device != model_device:
                if isinstance(batch, torch.Tensor):
                    return batch.to(model_device, non_blocking=True)
                elif isinstance(batch, (list, tuple)):
                    return [b.to(model_device, non_blocking=True) for b in batch]
            return batch

        if disable_tqdm is None:
            disable_tqdm = int(verbose) < 2

        node_sols = {}

        for NS in range(num_swipes):
            for node in self.train_nodes if NS % 2 == 0 else reversed(self.train_nodes):
                if timeout is not None and (time.time() - start_time) > timeout:
                    print(f"Timeout reached ({timeout} seconds). Stopping lanczos_swipe.")
                    return False

                # Precompute batches of HJ and b for this node
                b_rhs = torch.zeros_like(node.tensor)
                d_losss = []
                dd_losss = []
                loss_total = 0.0
                for b in range(batches):
                    x_batch = x[b*batch_size:(b+1)*batch_size] if isinstance(x, torch.Tensor) else [x[i][b*batch_size:(b+1)*batch_size] for i in range(len(x))]
                    y_batch = y_true[b*batch_size:(b+1)*batch_size]
                    x_batch = move_batch(x_batch)
                    y_batch = move_batch(y_batch)

                    y_pred = self.forward(x_batch, to_tensor=True)
                    loss, d_loss, sqd_loss = loss_fn.forward(y_pred, y_batch)

                    b_vec = self.get_b(node, d_loss)
                    b_rhs.add_(b_vec)

                    d_losss.append(d_loss)
                    dd_losss.append(sqd_loss)

                    if loss_callback is not None:
                        loss_total += loss.mean().item()
                if loss_callback is not None:
                    loss_callback(loss_total / batches)

                # Helper: matvec for this node (A @ v)
                t_bar = tqdm(total=max_iter, desc=f"Iterative pass ({node.name if hasattr(node, 'name') else 'node'})", disable=disable_tqdm)
                def matvec(v):
                    v = torch.tensor(v, dtype=b_rhs.dtype, device=b_rhs.device).reshape_as(b_rhs)
                    Av = 0
                    for b, d_loss, dd_loss in zip(range(batches), d_losss, dd_losss):
                        x_batch = x[b*batch_size:(b+1)*batch_size] if isinstance(x, torch.Tensor) else [x[i][b*batch_size:(b+1)*batch_size] for i in range(len(x))]
                        x_batch = move_batch(x_batch)


                        self.set_input(x_batch)
                        self.reset_stacks()
                        self.recompute_all_stacks()

                        prep_J = self.get_J(node, d_loss)
                        J = prep_J['J']
                        J_einsum = prep_J['einsum']
                        node_ein = prep_J['node_ein']
                        d_loss_ein = prep_J['d_loss_ein']
                        dd_loss_ein = prep_J['dd_loss_ein']
                        coeff_ein = prep_J['coeff_ein']
                        coeff = torch.einsum(f"{J_einsum},{node_ein},{dd_loss_ein}->{coeff_ein}", J.tensor, v, dd_loss)
                        Av += torch.einsum(f"{J_einsum},{d_loss_ein}->{node_ein}", J.tensor, coeff)
                    t_bar.update(1)
                    return Av.flatten().float().cpu().numpy()

                # Define the CG solver
                b_np = b_rhs.flatten().float().cpu().numpy()
                with torch.inference_mode():
                    A_op = LinearOperator((b_np.shape[0], b_np.shape[0]), matvec=matvec)
                    x_sol, info = solver(A_op, -b_np, x0=node_sols[node] if node in node_sols else None, maxiter=max_iter, rtol=tol)
                    node_sols[node] = x_sol
                step_tensor = torch.tensor(x_sol, dtype=b_rhs.dtype, device=b_rhs.device).reshape(b_rhs.shape)
                node.update_node(step_tensor, lr=lr)
                self.left_update_stacks(node)
                if block_callback is not None:
                    block_callback(NS, node)
                t_bar.close()
        return True


class CPDNetwork(TensorNetwork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.node_contract = None

    def set_input(self, x):
        """Sets the input tensor for the network."""
        was_updated = super().set_input(x)
        if was_updated:
            self.node_contract = None
        return was_updated

    def recompute_all_stacks(self):
        self.node_contract = {}
        for n in self.input_nodes:
            stack = n
            for vnode in self.get_column_nodes(n):
                stack = stack.contract_with(vnode)
            self.node_contract[n] = stack

    def compute_jacobian_stack(self, node) -> TensorNode:
        labeler = EinsumLabeler()
        nodes = [c if x not in self.get_column_nodes(node) else x for x, c in self.node_contract.items()]
        jac = torch.einsum(','.join([''.join([labeler[l] for l in n.dim_labels]) for n in nodes])+f'->{labeler[self.sample_dim]}'+''.join([labeler[l] for l in node.dim_labels if l in labeler.mapping]), *[n.tensor for n in nodes])
        return TensorNode(jac, dim_labels=[self.sample_dim]+[l for l in node.dim_labels if l in labeler.mapping], name='J')

    def forward(self, x, to_tensor=False):
        """Computes the forward pass of the network."""
        self.set_input(x)

        if self.node_contract is None:
            self.recompute_all_stacks()
        labeler = EinsumLabeler()
        out = torch.einsum(','.join([''.join([labeler[l] for l in n.dim_labels]) for n in self.node_contract.values()])+f'->{labeler[self.sample_dim]}'+''.join([labeler[l] for l in self.output_labels if l != self.sample_dim]), *[self.node_contract[n].tensor for n in self.input_nodes])
        node = TensorNode(out, dim_labels=[self.sample_dim] + [l for l in self.output_labels if l != self.sample_dim], name='O')
        if self.output_labels is not None:
            node = node.permute_first(*self.output_labels)
        if to_tensor:
            return node.tensor
        return node

    def reset_stacks(self, node=None):
        if node is not None:
            # Update the column corresponding to this node (first get the input node)
            input_node = next((n for n in self.input_nodes if n in self.get_column_nodes(node)), None)
            if input_node is not None:
                stack = input_node
                for vnode in self.get_column_nodes(input_node):
                    stack = stack.contract_with(vnode)
                self.node_contract[input_node] = stack
        else:
            self.node_contract = None

class SumOfNetworks(TensorNetwork):
    # A function which takes multiple tensor networks and sums their outputs.
    # We need to define the recompute_all_stacks, forward and reset_stacks methods.
    def __init__(self, networks, output_labels=('s',), sample_dim='s', train_operators=True):
        input_nodes = []
        main_nodes = []
        train_nodes = []
        for i, net in enumerate(networks, 1):
            for n in net.input_nodes:
                n.name = f"{n.name}_n{i}"
            input_nodes.extend(net.input_nodes)
            for n in net.main_nodes:
                n.name = f"{n.name}_n{i}"
            main_nodes.extend(net.main_nodes)
            if train_operators:
                train_nodes.extend(net.train_nodes)
            else:
                train_nodes.extend(net.main_nodes)
        super().__init__(input_nodes, main_nodes, train_nodes, output_labels=output_labels, sample_dim=sample_dim)
        self.networks = networks

    def forward(self, x, to_tensor=False):
        out = None
        for i, net in enumerate(self.networks):
            y = net.forward([x[..., *[slice(0, b.shape[i]) for i in range(1, b.tensor.ndim)]] for b in net.input_nodes], to_tensor=False)
            if self.output_labels is not None:
                y = y.permute_first(*self.output_labels)
            if out is None:
                out = y
            else:
                out.tensor = out.tensor + y.tensor
        if to_tensor:
            out = out.tensor
        return out

    def get_A_b(self, node, grad, hessian, method=None):
        for net in self.networks:
            if node in net.nodes:
                return net.get_A_b(node, grad, hessian)
        raise ValueError("Node not found in any network")
    
    def reset_stacks(self, node=None):
        for net in self.networks:
            if node in net.nodes:
                net.reset_stacks(node)

    def recompute_all_stacks(self):
        for net in self.networks:
            net.recompute_all_stacks()

    def orthonormalize_left(self):
        for net in self.networks:
            net.orthonormalize_left()

    def orthonormalize_right(self):
        for net in self.networks:
            net.orthonormalize_right()

    def node_orthonormalize_left(self, node):
        for net in self.networks:
            if node in net.main_nodes:
                net.node_orthonormalize_left(node)

    def node_orthonormalize_right(self, node):
        for net in self.networks:
            if node in net.main_nodes:
                net.node_orthonormalize_right(node)

    def left_update_stacks(self, node):
        raise NotImplementedError("left_update_stacks not implemented for SumOfNetworks")

    def right_update_stacks(self, node):
        raise NotImplementedError("right_update_stacks not implemented for SumOfNetworks")
