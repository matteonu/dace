# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" This module contains classes and functions that implement the orthogonal
    tiling transformation. """

import dace
from dace import registry, symbolic
from dace.properties import make_properties, Property, ShapeProperty
from dace.sdfg import nodes, propagation
from dace.sdfg import utils as sdutil
from dace.sdfg.sdfg import SDFG
from dace.sdfg.state import SDFGState
from dace.transformation import transformation
from dace.transformation import helpers



@make_properties
class MapTiling(transformation.SingleStateTransformation):
    """ Implements the orthogonal tiling transformation.

        Orthogonal tiling is a type of nested map fission that creates tiles
        in every dimension of the matched Map.
    """
    map_entry = transformation.PatternNode(nodes.MapEntry)

    # Properties
    prefix = Property(dtype=str, default="tile", desc="Prefix for new range symbols")
    tile_sizes = ShapeProperty(dtype=tuple, default=(128, 128, 128), desc="Tile size per dimension")

    strides = ShapeProperty(dtype=tuple,
                            default=tuple(),
                            desc="Tile stride (enables overlapping tiles). If empty, matches tile")

    tile_offset = ShapeProperty(dtype=tuple, default=None, desc="Negative Stride offset per dimension", allow_none=True)

    divides_evenly = Property(dtype=bool, default=False, desc="Tile size divides dimension length evenly")
    tile_trivial = Property(dtype=bool, default=False, desc="Tiles even if tile_size is 1")

    @staticmethod
    def annotates_memlets():
        return True

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.map_entry)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        return True

    def apply(self, graph: SDFGState, sdfg: SDFG):
        tile_strides = self.tile_sizes
        if self.strides is not None and len(self.strides) == len(tile_strides):
            tile_strides = self.strides

        # Retrieve map entry and exit nodes.
        map_entry = self.map_entry
        from dace.transformation.dataflow.map_collapse import MapCollapse
        from dace.transformation.dataflow.strip_mining import StripMining
        stripmine_subgraph = {StripMining.map_entry: self.subgraph[MapTiling.map_entry]}
        sdfg_id = sdfg.sdfg_id
        last_map_entry = None
        removed_maps = 0

        original_schedule = map_entry.schedule
        original_divides_evenly = self.divides_evenly

        if not self.divides_evenly:
            scope = graph.scope_subgraph(map_entry, True, True)
            removed_maps = 0
            self.divides_evenly =True

            import itertools
            #spawn residue maps and tile them if possible
            for i in range(0,len(map_entry.map.params)):
                combination = (i+1) * [True] + (len(map_entry.map.params) - (i + 1)) * [False]
                permutations = list(set(list(itertools.permutations(combination))))
                for permutation in permutations:
                    residue_map = helpers.replicate_scope(sdfg=sdfg, state=graph, scope=scope)
                    residue_map_entry = residue_map.entry
        
                    #change the ranges for each dimension
                    for dim_idx in range(len(residue_map_entry.map.params)):
                        only_residue = permutation[dim_idx]
                        td_from, td_to, td_step = residue_map_entry.map.range[dim_idx]

                        if dim_idx >= len(self.tile_sizes):
                            tile_size = symbolic.pystr_to_symbolic(self.tile_sizes[-1])
                            tile_stride = symbolic.pystr_to_symbolic(tile_strides[-1])
                        else:
                            tile_size = symbolic.pystr_to_symbolic(self.tile_sizes[dim_idx])
                            tile_stride = symbolic.pystr_to_symbolic(tile_strides[dim_idx])

                        td_to_new = td_to
                        td_from_new = td_from
                        if only_residue:
                            td_from_new = td_from + ((td_to + 1) - symbolic.sympy.Mod(((td_to + 1) - td_from), tile_size))
                        else:
                            td_to_new = td_from + ((td_to + 1) - symbolic.sympy.Mod(((td_to + 1) - td_from), tile_size)) - 1

                        residue_map_entry.map.range[dim_idx] = (td_from_new, td_to_new, td_step)

                    #apply tiling and leave out residue dimensions
                    residue_last_map_entry = None
                    stripmine_subgraph = {StripMining.map_entry: graph.node_id(residue_map_entry)}
                    removed_maps = 0
                    for dim_idx in range(len(residue_map_entry.map.params)):
                        only_residue = permutation[dim_idx]
                        if dim_idx >= len(self.tile_sizes):
                            tile_size = symbolic.pystr_to_symbolic(self.tile_sizes[-1])
                            tile_stride = symbolic.pystr_to_symbolic(tile_strides[-1])
                        else:
                            tile_size = symbolic.pystr_to_symbolic(self.tile_sizes[dim_idx])
                            tile_stride = symbolic.pystr_to_symbolic(tile_strides[dim_idx])

                        # handle offsets
                        if self.tile_offset and dim_idx >= len(self.tile_offset):
                            offset = self.tile_offset[-1]
                        elif self.tile_offset:
                            offset = self.tile_offset[dim_idx]
                        else:
                            offset = 0   

                        dim_idx -= removed_maps

                        if tile_size == residue_map_entry.map.range.size()[dim_idx] or only_residue:
                            continue
                        
                        #TODO: do i need to change the subgraph?
                        stripmine = StripMining()
                        stripmine.setup_match(sdfg, sdfg_id, self.state_id, stripmine_subgraph, self.expr_index)

                        # Special case: Tile size of 1 should be omitted from inner map
                        if tile_size == 1 and tile_stride == 1 and self.tile_trivial == False:
                            stripmine.dim_idx = dim_idx
                            stripmine.new_dim_prefix = ''
                            stripmine.tile_size = str(tile_size)
                            stripmine.tile_stride = str(tile_stride)
                            stripmine.divides_evenly = True
                            stripmine.tile_offset = str(offset)
                            stripmine.apply(graph, sdfg)
                            removed_maps += 1
                        else:
                            stripmine.dim_idx = dim_idx
                            stripmine.new_dim_prefix = self.prefix
                            stripmine.tile_size = str(tile_size)
                            stripmine.tile_stride = str(tile_stride)
                            stripmine.divides_evenly = self.divides_evenly
                            stripmine.tile_offset = str(offset)
                            stripmine.apply(graph, sdfg)
                        
                        # apply to the new map the schedule of the original one
                        residue_map_entry.schedule = original_schedule

                        if residue_last_map_entry:
                            new_map_entry = graph.in_edges(residue_map_entry)[0].src
                            mapcollapse_subgraph = {
                                MapCollapse.outer_map_entry: graph.node_id(residue_last_map_entry),
                                MapCollapse.inner_map_entry: graph.node_id(new_map_entry)
                            }
                            mapcollapse = MapCollapse()
                            mapcollapse.setup_match(sdfg, sdfg_id, self.state_id, mapcollapse_subgraph, 0)
                            mapcollapse.apply(graph, sdfg)
                        residue_last_map_entry = graph.in_edges(residue_map_entry)[0].src

            # adapt the upper bound of the original map such that it only operates on the biggest evenly divisible chunk of data
            for dim_idx in range(len(map_entry.map.params)):
                td_from, td_to, td_step = map_entry.map.range[dim_idx]

                if dim_idx >= len(self.tile_sizes):
                    tile_size = symbolic.pystr_to_symbolic(self.tile_sizes[-1])
                    tile_stride = symbolic.pystr_to_symbolic(tile_strides[-1])
                else:
                    tile_size = symbolic.pystr_to_symbolic(self.tile_sizes[dim_idx])
                    tile_stride = symbolic.pystr_to_symbolic(tile_strides[dim_idx])

                td_to_new = td_from + ((td_to + 1) - symbolic.sympy.Mod(((td_to + 1) - td_from), tile_size)) - 1
                map_entry.map.range[dim_idx] = (td_from, td_to_new, td_step)

            propagation.propagate_memlets_sdfg(sdfg)
            removed_maps = 0
            stripmine_subgraph = {StripMining.map_entry: self.subgraph[MapTiling.map_entry]}
            self.divides_evenly = True


        for dim_idx in range(len(map_entry.map.params)):
            if dim_idx >= len(self.tile_sizes):
                tile_size = symbolic.pystr_to_symbolic(self.tile_sizes[-1])
                tile_stride = symbolic.pystr_to_symbolic(tile_strides[-1])
            else:
                tile_size = symbolic.pystr_to_symbolic(self.tile_sizes[dim_idx])
                tile_stride = symbolic.pystr_to_symbolic(tile_strides[dim_idx])

            # handle offsets
            if self.tile_offset and dim_idx >= len(self.tile_offset):
                offset = self.tile_offset[-1]
            elif self.tile_offset:
                offset = self.tile_offset[dim_idx]
            else:
                offset = 0

            dim_idx -= removed_maps
            # If tile size is trivial, skip strip-mining map dimension
            if tile_size == map_entry.map.range.size()[dim_idx]:
                continue

            stripmine = StripMining()
            stripmine.setup_match(sdfg, sdfg_id, self.state_id, stripmine_subgraph, self.expr_index)

            # Special case: Tile size of 1 should be omitted from inner map
            if tile_size == 1 and tile_stride == 1 and self.tile_trivial == False:
                stripmine.dim_idx = dim_idx
                stripmine.new_dim_prefix = ''
                stripmine.tile_size = str(tile_size)
                stripmine.tile_stride = str(tile_stride)
                stripmine.divides_evenly = True
                stripmine.tile_offset = str(offset)
                stripmine.apply(graph, sdfg)
                removed_maps += 1
            else:
                stripmine.dim_idx = dim_idx
                stripmine.new_dim_prefix = self.prefix
                stripmine.tile_size = str(tile_size)
                stripmine.tile_stride = str(tile_stride)
                stripmine.divides_evenly = self.divides_evenly
                stripmine.tile_offset = str(offset)
                stripmine.apply(graph, sdfg)

            # apply to the new map the schedule of the original one
            map_entry.schedule = original_schedule

            if last_map_entry:
                new_map_entry = graph.in_edges(map_entry)[0].src
                mapcollapse_subgraph = {
                    MapCollapse.outer_map_entry: graph.node_id(last_map_entry),
                    MapCollapse.inner_map_entry: graph.node_id(new_map_entry)
                }
                mapcollapse = MapCollapse()
                mapcollapse.setup_match(sdfg, sdfg_id, self.state_id, mapcollapse_subgraph, 0)
                mapcollapse.apply(graph, sdfg)
            last_map_entry = graph.in_edges(map_entry)[0].src

        self.divides_evenly = original_divides_evenly
        return last_map_entry
    

