# The original code is from https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.RGCNConv.html
# It is modified to not only include bases
# but include bases in the following sense:
# 1. num_m -> number of bases shared amongst all
# 2. num_n -> num bases shared between source-target and target-source graph per relation type
# 3. num_o -> num bases not shared with any other relation type

import torch
from torch import Tensor
from torch.nn import Parameter
from torch.nn import Parameter as Param
import torch
import torch.nn.functional as F
from torch_geometric.nn import Linear
from torch_geometric.utils import scatter
from typing import Optional, Tuple, Union
import torch_geometric.backend
import torch_geometric.typing
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import (
    Adj,
    OptTensor,
    SparseTensor,
    pyg_lib,
    torch_sparse,
)
from torch_geometric.utils import index_sort, one_hot, scatter, spmm
from torch_geometric.utils.sparse import index2ptr

@torch.jit._overload
def masked_edge_index(edge_index, edge_mask):
    # type: (Tensor, Tensor) -> Tensor
    pass


@torch.jit._overload
def masked_edge_index(edge_index, edge_mask):
    # type: (SparseTensor, Tensor) -> SparseTensor
    pass


def masked_edge_index(edge_index, edge_mask):
    if isinstance(edge_index, Tensor):
        return edge_index[:, edge_mask]
    return torch_sparse.masked_select_nnz(edge_index, edge_mask, layout='coo')


class DRGCNConv(MessagePassing):
    r"""The relational graph convolutional operator from the `"Modeling
    Relational Data with Graph Convolutional Networks"
    <https://arxiv.org/abs/1703.06103>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}_{\textrm{root}} \cdot
        \mathbf{x}_i + \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_r(i)}
        \frac{1}{|\mathcal{N}_r(i)|} \mathbf{\Theta}_r \cdot \mathbf{x}_j,

    where :math:`\mathcal{R}` denotes the set of relations, *i.e.* edge types.
    Edge type needs to be a one-dimensional :obj:`torch.long` tensor which
    stores a relation identifier
    :math:`\in \{ 0, \ldots, |\mathcal{R}| - 1\}` for each edge.

    .. note::
        This implementation is as memory-efficient as possible by iterating
        over each individual relation type.
        Therefore, it may result in low GPU utilization in case the graph has a
        large number of relations.
        As an alternative approach, :class:`FastDRGCNConv` does not iterate over
        each individual type, but may consume a large amount of memory to
        compensate.
        We advise to check out both implementations to see which one fits your
        needs.

    .. note::
        :class:`DRGCNConv` can use `dynamic shapes
        <https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index
        .html#work_dynamic_shapes>`_, which means that the shape of the interim
        tensors can be determined at runtime.
        If your device doesn't support dynamic shapes, use
        :class:`FastDRGCNConv` instead.

    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
            In case no input features are given, this argument should
            correspond to the number of nodes in your graph.
        out_channels (int): Size of each output sample.
        num_relations (int): Number of relations.
        num_bases (int, optional): If set, this layer will use the
            basis-decomposition regularization scheme where :obj:`num_bases`
            denotes the number of bases to use. (default: :obj:`None`)
        num_blocks (int, optional): If set, this layer will use the
            block-diagonal-decomposition regularization scheme where
            :obj:`num_blocks` denotes the number of blocks to use.
            (default: :obj:`None`)
        aggr (str, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"mean"`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        is_sorted (bool, optional): If set to :obj:`True`, assumes that
            :obj:`edge_index` is sorted by :obj:`edge_type`. This avoids
            internal re-sorting of the data and can improve runtime and memory
            efficiency. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        num_relations: int,
        # num_bases: Optional[int] = None,
        num_m: Optional[int] = None,
        num_n: Optional[int] = None,
        num_o: Optional[int] = None,
        num_blocks: Optional[int] = None,
        aggr: str = 'mean',
        root_weight: bool = True,
        is_sorted: bool = False,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', aggr)
        super().__init__(node_dim=0, **kwargs)

        self.is_num_b = num_m is not None or num_n is not None or num_o is not None
        if self.is_num_b and num_blocks is not None:
            raise ValueError('Can not apply both basis-decomposition and '
                             'block-diagonal-decomposition at the same time.')

        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.num_m = num_m # all share these
        self.num_relations = num_relations*2
        self.num_actual_relations = num_relations
        self.num_n_step = num_n
        self.num_n = num_n * num_relations # n per relation type
        self.num_o_step = num_o
        self.num_o = num_o * num_relations * 2 # nper relation type  per direction 
        self.num_blocks = num_blocks
        self.is_sorted = is_sorted

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        self.in_channels_l = in_channels[0]

        self._use_segment_matmul_heuristic_output: Optional[bool] = None

        if self.is_num_b:
            self.num_bases = self.num_m + self.num_n + self.num_o
            self.weight = Parameter(
                torch.empty(self.num_bases, in_channels[0], out_channels)) # (num_b, in, out)
            self.comp = Parameter(torch.empty(self.num_relations, self.num_bases)) # num_r, num_b
            self.weight_mask = torch.zeros(self.num_relations, self.num_bases, dtype=torch.bool).requires_grad_(False)

            self.weight_mask[:, :self.num_m] = 1 # all share the same m bases

            for row_i in range(self.num_relations):
                for col_i in range(0,self.num_actual_relations):
                    # set 1 for every relationship type (both directions have 1 in same column)
                    if row_i == col_i or row_i == col_i + self.num_actual_relations:
                        col_i = col_i * self.num_n_step
                        self.weight_mask[row_i, self.num_m+col_i: self.num_m+col_i+self.num_n_step] = 1
                        
                # set 1 per relationship type per direction
                for col_i in range(self.num_relations):
                    if row_i == col_i:
                        start = self.num_m + self.num_n + col_i * self.num_o_step
                        end = self.num_m + self.num_n + (col_i+1) * self.num_o_step
                        self.weight_mask[row_i, start:end] = 1
            print(self.weight_mask.int())
            
        elif num_blocks is not None:
            assert (in_channels[0] % num_blocks == 0
                    and out_channels % num_blocks == 0)
            self.weight = Parameter(
                torch.empty(num_relations, num_blocks,
                            in_channels[0] // num_blocks,
                            out_channels // num_blocks))
            self.register_parameter('comp', None)

        else:
            self.weight = Parameter(
                torch.empty(num_relations, in_channels[0], out_channels))
            self.register_parameter('comp', None)

        if root_weight:
            self.root = Param(torch.empty(in_channels[1], out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Param(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        glorot(self.weight)
        glorot(self.comp)
        glorot(self.root)
        zeros(self.bias)

    def forward(self, x: Union[OptTensor, Tuple[OptTensor, Tensor]],
                edge_index: Adj, edge_type: OptTensor = None):
        r"""Runs the forward pass of the module.

        Args:
            x (torch.Tensor or tuple, optional): The input node features.
                Can be either a :obj:`[num_nodes, in_channels]` node feature
                matrix, or an optional one-dimensional node index tensor (in
                which case input features are treated as trainable node
                embeddings).
                Furthermore, :obj:`x` can be of type :obj:`tuple` denoting
                source and destination node features.
            edge_index (torch.Tensor or SparseTensor): The edge indices.
            edge_type (torch.Tensor, optional): The one-dimensional relation
                type/index for each edge in :obj:`edge_index`.
                Should be only :obj:`None` in case :obj:`edge_index` is of type
                :class:`torch_sparse.SparseTensor`. (default: :obj:`None`)
        """
        
        # add edges in other direction
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_type  = torch.cat([edge_type, edge_type + self.num_actual_relations], dim=0)
        
        # Convert input features to a pair of node features or node indices.
        x_l: OptTensor = None
        if isinstance(x, tuple):
            x_l = x[0]
        else:
            x_l = x
        if x_l is None:
            x_l = torch.arange(self.in_channels_l, device=self.weight.device)

        x_r: Tensor = x_l
        if isinstance(x, tuple):
            x_r = x[1]

        size = (x_l.size(0), x_r.size(0))
        if isinstance(edge_index, SparseTensor):
            edge_type = edge_index.storage.value()
        assert edge_type is not None

        # propagate_type: (x: Tensor, edge_type_ptr: OptTensor)
        out = torch.zeros(x_r.size(0), self.out_channels, device=x_r.device)


        # (num_b, in, out)
        # self.comp = Parameter(torch.empty(num_relations, num_bases)) # num_r, num_b
            
        weight = self.weight  #  (num_b, in, out)
      
        if self.num_bases is not None:   # Basis-decomposition =================
            self.weight_mask = self.weight_mask.to(self.comp.device)
            weight = (torch.mul(self.weight_mask,self.comp) @ weight.view(self.num_bases, -1)).view(
                self.num_relations, self.in_channels_l, self.out_channels)


        if self.num_blocks is not None:  # Block-diagonal-decomposition =====

            if not torch.is_floating_point(
                    x_r) and self.num_blocks is not None:
                raise ValueError('Block-diagonal decomposition not supported '
                                 'for non-continuous input features.')

            for i in range(self.num_relations):
                tmp = masked_edge_index(edge_index, edge_type == i)
                h = self.propagate(tmp, x=x_l, edge_type_ptr=None, size=size)
                h = h.view(-1, weight.size(1), weight.size(2))
                h = torch.einsum('abc,bcd->abd', h, weight[i])
                out = out + h.contiguous().view(-1, self.out_channels)

        else:  # No regularization/Basis-decomposition ========================

            use_segment_matmul = torch_geometric.backend.use_segment_matmul
            # If `use_segment_matmul` is not specified, use a simple heuristic
            # to determine whether `segment_matmul` can speed up computation
            # given the observed input sizes:
            if use_segment_matmul is None:
                segment_count = scatter(torch.ones_like(edge_type), edge_type,
                                        dim_size=self.num_relations)

                self._use_segment_matmul_heuristic_output = (
                    torch_geometric.backend.use_segment_matmul_heuristic(
                        num_segments=self.num_relations,
                        max_segment_size=int(segment_count.max()),
                        in_channels=self.weight.size(1),
                        out_channels=self.weight.size(2),
                    ))

                assert self._use_segment_matmul_heuristic_output is not None
                use_segment_matmul = self._use_segment_matmul_heuristic_output

            if (use_segment_matmul and torch_geometric.typing.WITH_SEGMM
                    and self.num_bases is None and x_l.is_floating_point()
                    and isinstance(edge_index, Tensor)):

                if not self.is_sorted:
                    if (edge_type[1:] < edge_type[:-1]).any():
                        edge_type, perm = index_sort(
                            edge_type, max_value=self.num_relations)
                        edge_index = edge_index[:, perm]
                edge_type_ptr = index2ptr(edge_type, self.num_relations)
                out = self.propagate(edge_index, x=x_l,
                                     edge_type_ptr=edge_type_ptr, size=size)
            else:
                for i in range(self.num_relations):
                    tmp = masked_edge_index(edge_index, edge_type == i)

                    if not torch.is_floating_point(x_r):
                        out = out + self.propagate(
                            tmp,
                            x=weight[i, x_l],
                            edge_type_ptr=None,
                            size=size,
                        )
                    else:
                        h = self.propagate(tmp, x=x_l, edge_type_ptr=None,
                                           size=size)
                        out = out + (h @ weight[i])

        root = self.root
        if root is not None:
            if not torch.is_floating_point(x_r):
                out = out + root[x_r]
            else:
                out = out + x_r @ root

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j: Tensor, edge_type_ptr: OptTensor) -> Tensor:
        if torch_geometric.typing.WITH_SEGMM and edge_type_ptr is not None:
            # TODO Re-weight according to edge type degree for `aggr=mean`.
            return pyg_lib.ops.segment_matmul(x_j, edge_type_ptr, self.weight)

        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        adj_t = adj_t.set_value(None)
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_relations={self.num_relations})')
        



class BareboneDRGCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, dropout, readout_pooling, num_m, num_n, num_o,aggr='mean', **kwargs):
        """
            num_m, num_n, num_o: see DRGCNConv
        """
        super().__init__()
        
        self.dropout = dropout
        self.pooling = readout_pooling
        self.layers = torch.nn.ModuleList()
        
        self.num_relations=5 #
   
        
        self.initial_layer = DRGCNConv(in_channels=162, out_channels=hidden_channels,num_relations=self.num_relations,num_m=num_m, num_n=num_n, num_o=num_o, is_sorted=True, aggr=aggr)
        
        self.layers = torch.nn.ModuleList()
        for _ in range(num_layers-1):
            self.layers.append(DRGCNConv(in_channels=hidden_channels, out_channels=hidden_channels,num_relations=self.num_relations,num_m=num_m, num_n=num_n, num_o=num_o, is_sorted=True, aggr=aggr))
        
        self.classifcation_layer1 = Linear(hidden_channels, hidden_channels)
        self.classifcation_layer2 = Linear(hidden_channels, hidden_channels)
        self.classification_layer3 = Linear(hidden_channels, 1)
            

    def forward(self, X, edge_index1, edge_index2, edge_index3, edge_index4, edge_index5, batch):
        tensorsandedgetypes = [(tensor, i*torch.ones(tensor.shape[1], device=X.device)) for i, tensor in enumerate([edge_index1, edge_index2, edge_index3, edge_index4, edge_index5]) if tensor.nelement() > 0]
        edge_index = torch.concatenate([v[0] for v in tensorsandedgetypes], dim=1).to(torch.long)
        edge_type = torch.concatenate([v[1] for v in tensorsandedgetypes], dim=0).to(torch.long)
        
        X = self.initial_layer(X, edge_index, edge_type)
        X = F.dropout(X, training=self.training, p=self.dropout)
        X = F.relu(X)
        for layer in self.layers:
            X = layer(x=X, edge_index=edge_index, edge_type=edge_type)
            X = F.dropout(X, training=self.training, p=self.dropout)
            X = F.relu(X)
        
        
        read_out = scatter(src=X, index=batch, dim=0, reduce=self.pooling)
        X = F.relu(self.classifcation_layer1(read_out))
        X = F.relu(self.classifcation_layer2(X))
        logits = self.classification_layer3(X)

        return logits
    
if __name__ == '__main__':
    pass
    
    