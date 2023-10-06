# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.

from collections import defaultdict
from typing import Any, Dict, Optional, Set, Tuple, List
import networkx as nx

from dace import SDFG
from dace import subsets
from dace.sdfg import nodes as nd
from dace.sdfg import SDFGState
from dace.memlet import Memlet
from dace.transformation import pass_pipeline as ppl
from dace.sdfg.graph import Edge
from dace.sdfg.propagation_underapproximation import UnderapproximateWrites
from dace.transformation.passes import analysis as ap


class _PhiNode():
    def __init__(self, name: str, variables: set[str]):
        self.name: str = name
        self.variables: set[str] = variables


class ArrayFission(ppl.Pass):
    """
    Fission transient arrays that are dominated by full writes 
    to the whole array into separate data containers.
    """

    CATEGORY: str = 'Optimization Preparation'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.AccessNodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & ppl.Modifies.AccessNodes

    def depends_on(self):
        return {UnderapproximateWrites,
                ap.AccessSets,
                ap.FindAccessNodes,
                ap.StateReachability}

    def apply_pass(
        self,
        sdfg: SDFG,
        pipeline_results: Dict[str, Any]
    ) -> Optional[Dict[str, Set[str]]]:
        """
        Rename scalars and arrays of size 1 based on dominated scopes.

        :param sdfg: The SDFG to modify.
        :param pipeline_results: If in the context of a ``Pipeline``, a dictionary that is 
                                populated with prior Pass results as ``{Pass subclass name: 
                                returned object from pass}``. If not run in a pipeline, an 
                                empty dictionary is expected.
        :return: A dictionary mapping the original name to a set of all new names created 
                for each data container.
        """
        results: Dict[str, Set[str]] = defaultdict(set)
        write_approximation: dict[Edge,
                                  Memlet] = pipeline_results[UnderapproximateWrites.__name__]["approximation"]
        loop_write_approximation: Dict[SDFGState, Dict[str, Memlet]
                                       ] = pipeline_results[UnderapproximateWrites.__name__]["loop_approximation"]
        loops: Dict[SDFGState, Tuple[SDFGState, SDFGState, List[SDFGState], str, subsets.Range]
                    ] = pipeline_results[UnderapproximateWrites.__name__]["loops"]
        access_nodes: Dict[str, Dict[SDFGState, Tuple[Set[nd.AccessNode], Set[nd.AccessNode]]]
                           ] = pipeline_results[ap.FindAccessNodes.__name__][sdfg.sdfg_id]
        state_reach: Dict[SDFGState, Set[SDFGState]
                          ] = pipeline_results[ap.StateReachability.__name__][sdfg.sdfg_id]
        access_sets: Dict[SDFGState, Tuple[Set[str], Set[str]]
                          ] = pipeline_results[ap.AccessSets.__name__][sdfg.sdfg_id]
        # list of original array names in the sdfg that fissioning is performed on
        array_variable_names: List[str] = [aname for aname, a in sdfg.arrays.items(
        ) if a.transient and not a.total_size == 1]

        # dictionary that stores "virtual" phi nodes for each variable and SDFGstate
        # phi nodes are represented by dictionaries that contain:
        # - the variable defined by the phi-node
        # - the corresponding descriptor
        # - the definitions reaching the phi node
        # can be extended for later use with path constraints for example
        phi_nodes: Dict[SDFGState, Dict[str, _PhiNode]] = defaultdict(None)

        # maps each variable to it's defining writes that don't involve phi-nodes
        def_states: Dict[str, Set[SDFGState]] = {}

        # maps each variable to it's defining writes that involve phi-nodes
        def_states_phi: Dict[str, Set[SDFGState]] = {}

        phi_nodes, def_states, def_states_phi = _insert_phi_nodes(
            sdfg,
            array_variable_names,
            write_approximation,
            loop_write_approximation,
            access_nodes,
        )

        _rename(
            sdfg,
            write_approximation,
            phi_nodes,
            array_variable_names
        )

        _eliminate_phi_nodes(
            sdfg,
            phi_nodes,
            def_states,
            def_states_phi,
            loops,
            loop_write_approximation,
            state_reach,
            access_nodes,
            access_sets
        )

        # store all the new names for each original variable name
        definitions: Dict[str, Set(str)] = defaultdict(None)
        for var, an_dict in access_nodes.items():
            current_defs = set()
            for _, ans in an_dict.items():
                ans = ans[0].union(ans[1])
                for an in ans:
                    current_defs.add(an.data)
            if len(current_defs) > 1:
                definitions[var] = current_defs
        results = definitions
        return results

    def report(self, pass_retval: Any) -> Optional[str]:
        return f'Renamed {len(pass_retval)} arrays: {pass_retval}.'


def _insert_phi_nodes(
    sdfg: SDFG,
    array_variable_names: List[str],
    write_approximation: dict[Edge, Memlet],
    loop_write_approximation: dict[SDFGState, dict[str, Memlet]],
    access_nodes: Dict[str, Dict[SDFGState,
                                 Tuple[Set[nd.AccessNode], Set[nd.AccessNode]]]]
) -> Tuple[Dict[SDFGState, Dict[str, _PhiNode]],
           Dict[str, set[SDFGState]],
           Dict[str, set[SDFGState]]]:

    phi_nodes: Dict[SDFGState, Dict[str, _PhiNode]] = defaultdict(None)
    for state in sdfg.states():
        phi_nodes[state] = {}
    # maps each variable to its defining writes that don't involve phi-nodes
    defining_states: Dict[str, set[SDFGState]] = {}
    # maps each variable to its defining writes that involve phi-nodes
    defining_states_phi: Dict[str, set[SDFGState]] = {}

    defining_states_phi = _insert_phi_nodes_loopheaders(
        sdfg, array_variable_names, loop_write_approximation, phi_nodes)

    defining_states = _find_defining_states(
        sdfg, array_variable_names, phi_nodes, access_nodes, write_approximation)

    _insert_phi_nodes_regular(sdfg, defining_states,
                              phi_nodes, defining_states_phi)

    return (phi_nodes, defining_states, defining_states_phi)


def _rename(
        sdfg: SDFG,
        write_approximation: Dict[Edge, Memlet],
        phi_nodes: Dict[SDFGState, Dict[str, _PhiNode]],
        array_variable_names: List[str]
):
    # dictionary mapping each variable from the original SDFG to a dictionary mapping
    # each state to the last definition of that variable in that state
    last_defs: Dict[str, Dict[SDFGState, str]] = defaultdict(None)
    immediate_dominators = nx.dominance.immediate_dominators(
        sdfg.nx, sdfg.start_state)
    # traverse the dominator tree depth first and rename all variables
    dom_tree_dfs = _dominator_tree_DFS_order(sdfg, immediate_dominators)
    for current_state in dom_tree_dfs:
        _rename_DFG_and_interstate_edges(sdfg, current_state, array_variable_names, phi_nodes, write_approximation,
                                         last_defs, immediate_dominators)
        _propagate_new_names_to_phi_nodes(sdfg, current_state,
                                          array_variable_names, phi_nodes, last_defs)


def _eliminate_phi_nodes(
    sdfg: SDFG,
    phi_nodes: Dict[SDFGState, Dict[str, _PhiNode]],
    def_states: Dict[str, Set[SDFGState]],
    def_states_phi: Dict[str, Set[SDFGState]],
    loops: Dict[SDFGState, Tuple[SDFGState, SDFGState, List[SDFGState], str, subsets.Range]],
    loop_write_approximation: Dict[SDFGState, Dict[str, Memlet]],
    state_reach: Dict[SDFGState, Set[SDFGState]],
    access_nodes: Dict[str, Dict[SDFGState, Tuple[Set[nd.AccessNode], Set[nd.AccessNode]]]],
    access_sets: Dict[SDFGState, Tuple[Set[str], Set[str]]]
):

    # dictionary mapping each variable in the SDFG to the states that read from it.
    # For reads in edges only take outgoing edges into account
    var_reads: Dict[str, Set[SDFGState]] = defaultdict(set)
    # Initialize var_reads
    for state in sdfg.nodes():
        for anode in state.data_nodes():
            if state.out_degree(anode) > 0:
                var_reads[anode.data].add(state)
    array_names = sdfg.arrays.keys()
    for e in sdfg.edges():
        fsyms = e.data.free_symbols & array_names
        for access in fsyms:
            var_reads[access].update({e.src})

    # iterate over the phi nodes and replace all the occurences of each parameter with the
    # variable defined by the phi node
    to_remove: Set[str] = set()
    for state in sdfg.states():
        if not state in phi_nodes:
            continue
        phi_dict = phi_nodes[state]
        for original_var, phi_node in phi_dict.items():
            # name of variable defined by phi node
            newname = phi_node.name
            # variables that can reach phi-node/parameters of phi-node
            parameters = phi_node.variables

            # Find all states that can be reached by newname
            reached_by_def = _conditional_dfs(
                sdfg,
                state,
                lambda node: (
                    node is state
                    or (node not in def_states_phi[original_var]
                        and node not in def_states[original_var])
                )
            )

            candidate_states = sdfg.states()
            loop_states = []
            is_read = True  # is the variable defined by the phi-node read in the graph
            # if the state is a loop guard, does the loop overwrite the variable
            overwriting_loop = False

            # if the phi node belongs to a loopheader that completely overwrites the array
            # and the loop does not read from the array defined by the phi node
            # only rename nodes reached by the loopheader
            if (
                state in loop_write_approximation
                and original_var in loop_write_approximation[state]
                and loop_write_approximation[state][original_var].subset.covers_precise(
                    subsets.Range.from_array(sdfg.arrays[original_var])
                )
            ):
                _, _, loop_states, _, _ = loops[state]
                # check if loop reads from outside the loop
                if not any(
                        newname in [
                            a.label for a in access_nodes[original_var][other_state][0]]
                        or other_state in var_reads[newname]
                        for other_state in loop_states):

                    candidate_states = state_reach[state]
                    overwriting_loop = True
            # if the variable defined by the phi node is read by any other
            # state we perform renaming in the whole SDFG
            # if not we only perform phi propagation in all the states that are reachable by the defined variable
            elif not any(other_state in state_reach[state] or other_state is state for other_state in var_reads[newname]):
                candidate_states = reached_by_def
                is_read = False

            # rename phi nodes and propagate parameters
            _rename_phi_related_phi_nodes(parameters, original_var, newname, candidate_states,
                                          state, phi_nodes, reached_by_def, is_read, overwriting_loop)
            # rename accesses
            if is_read:
                _rename_phi_related_accesses(sdfg, parameters, original_var, newname, candidate_states,
                                             loop_states, access_nodes, access_sets, overwriting_loop)

            # update var_read if any renaming was done
            if is_read:
                # if a state read from the parameter it now reads from the variable
                # defined by the current phi node
                for parameter in parameters:
                    if parameter is newname:
                        continue
                    read_remove = set()
                    for other_state in var_reads[parameter]:
                        if other_state in candidate_states:
                            var_reads[newname].update({other_state})
                            read_remove.add(other_state)
                    var_reads[parameter].difference_update(read_remove)

                    # try to remove renamed parameters from SDFG
                    to_remove.add(parameter)

            # remove the phi-node if it is not an overwriting loop.
            # If the latter is the case we keep it such that
            # no definition is propagated past this phi node
            if not overwriting_loop:
                def_states_phi[original_var].remove(state)

        del phi_nodes[state]
    for parameter in to_remove:
        try:
            sdfg.remove_data(parameter)
        except ValueError:
            continue


def _rename_phi_related_accesses(
    sdfg: SDFG,
    phi_node_parameters: set[str],
    original_variable: str,
    new_name: str,
    candidate_states: set[SDFGState],
    loop_states: set[SDFGState],
    access_nodes: Dict[str, Dict[SDFGState, Tuple[Set[nd.AccessNode], Set[nd.AccessNode]]]],
    access_sets: Dict[SDFGState, Tuple[Set[str], Set[str]]],
    overwriting_loop: bool
):
    rename_dict = {}
    for parameter in phi_node_parameters:
        rename_dict[parameter] = new_name
    for other_state in candidate_states:
        if overwriting_loop and other_state not in loop_states:
            continue
        # rename all accesses to the parameters by accessnodes
        reads, writes = access_nodes[original_variable][other_state]
        ans = reads.union(writes)
        for an in ans:
            if not an.data in phi_node_parameters:
                continue
            _rename_node(other_state, an, new_name)

        # rename all accesses to the parameters by interstate edges
        other_accesses = access_sets[other_state]
        if original_variable in other_accesses[0]:
            for oedge in sdfg.out_edges(other_state):
                oedge.data.replace_dict(rename_dict)


def _rename_phi_related_phi_nodes(
    phi_node_parameters: set[str],
    original_variable: str,
    new_name: str,
    candidate_states: set[SDFGState],
    phi_node_origin_state: SDFGState,
    phi_nodes: Dict[SDFGState, Dict[str, _PhiNode]],
    reached_by_def: set[SDFGState],
    is_read: bool,
    is_overwriting_loop: bool
):
    for other_state in candidate_states:
        # rename all phi nodes and propagate
        if (not other_state is phi_node_origin_state and
            other_state in phi_nodes and
                original_variable in phi_nodes[other_state]):

            other_phi_node = phi_nodes[other_state][original_variable]
            new_variables = set()

            # if the variable defined by the other phi-node is in the parameters
            # rename the variable
            if other_phi_node.name in phi_node_parameters and is_read:
                other_phi_node.name = new_name

            # propagate parameter or variable defined by phi node to other phi nodes
            # that can be reached by the definition
            if other_state in reached_by_def:
                if not is_read:
                    new_variables.update(phi_node_parameters)
                else:
                    new_variables.add(new_name)
            new_variables.update(other_phi_node.variables)
            other_phi_node.variables = new_variables
            phi_nodes[other_state][original_variable] = other_phi_node


def _update_last_def(
        state: SDFGState,
        new_def: str,
        original_name: str,
        last_defs:  Dict[str, Dict[SDFGState, str]]):
    if not last_defs.get(original_name):
        last_defs[original_name] = {}
    last_defs[original_name][state] = new_def


def _find_reaching_def(
        state: SDFGState,
        var: str, last_defs: Dict[str, Dict[SDFGState, str]],
        phi_nodes: Dict[SDFGState, Dict[str, _PhiNode]],
        immediate_dominators: Dict[SDFGState, SDFGState]):
    # helper function to find the reaching definition given an AccessNode and a state in the
    # original SDFG

    if var in last_defs and state in last_defs[var]:
        return last_defs[var][state]
    # if there is a phi node for the variable in the same state, return the variable
    # name defined by the phi node
    if phi_nodes[state].get(var):
        return phi_nodes[state][var].name
    # otherwise return the last definition of the immediate dominator
    idom = immediate_dominators[state]
    if var in last_defs and idom in last_defs[var]:
        return last_defs[var][idom]
    # in case the state is the initial state and there isn't any reaching definition
    # in the current state just return the original variable
    return var


def _rename_DFG_and_interstate_edges(
        sdfg: SDFG,
        state: SDFGState,
        array_variable_names: List[str],
        phi_nodes: Dict[SDFGState, Dict[str, _PhiNode]],
        write_approximation:  Dict[Edge, Memlet],
        last_defs: Dict[str, Dict[SDFGState, str]],
        immediate_dominators: Dict[SDFGState, SDFGState]
):

    for var in array_variable_names:
        # check if there is a phi node for the current variable in the current state
        if not phi_nodes[state].get(var):
            continue
        newdesc = sdfg.arrays[var].clone()
        newname = sdfg.add_datadesc(var, newdesc, find_new_name=True)
        phi_nodes[state][var].name = newname
        _update_last_def(state, newname, var, last_defs)
    # rename data nodes
    # get topological ordering of nodes in the dataflowgraph
    renamed = defaultdict(None)
    toposort = list(nx.topological_sort(state.nx))
    for node in toposort:
        if not isinstance(node, nd.AccessNode):
            continue
        var = node.data
        if var not in array_variable_names:
            continue
        # if array is not fully overwritten at this access node treat it as a use
        # otherwise as a def also make sure that this is the first (and last)
        # renaming in this state
        array = sdfg.arrays[var]
        array_set = subsets.Range.from_array(array)
        iedges = state.in_edges(node)
        if (any(write_approximation[edge].subset.covers_precise(array_set) for edge in iedges) and
                not renamed.get(var)):
            # rename the variable to the reaching definition
            newdesc = array.clone()
            newname = sdfg.add_datadesc(
                var, newdesc, find_new_name=True)
            _update_last_def(state, newname, var, last_defs)
            renamed[var] = True
        else:
            newname = _find_reaching_def(
                state, var, last_defs, phi_nodes, immediate_dominators)
        _rename_node(state, node, newname)
    # define last definition in this state if it has not been defined yet
    for var in array_variable_names:
        if not last_defs.get(var):
            last_defs[var] = {}
        if not last_defs[var].get(state):
            last_defs[var][state] = _find_reaching_def(
                state, var, last_defs, phi_nodes, immediate_dominators)
    # iterate over all the outgoing interstate edges of the current state and
    # rename all the occurences of the original variable to the last definition
    # in the current state
    rename_dict = {}
    for var in last_defs.keys():
        rename_dict[var] = last_defs[var][state]
    for oedge in sdfg.out_edges(state):
        oedge.data.replace_dict(rename_dict)


def _propagate_new_names_to_phi_nodes(
        sdfg: SDFG,
        state: SDFGState,
        array_variable_names: List[str],
        phi_nodes: Dict[SDFGState, Dict[str, _PhiNode]],
        last_defs: Dict[str, Dict[SDFGState, str]]):
    # propagate last definition in state to phi nodes in successor states
    successors = [edge.dst for edge in sdfg.out_edges(state)]
    for successor in successors:
        for var in array_variable_names:
            if not phi_nodes[successor].get(var):
                continue
            newname = last_defs[var][state]
            phi_nodes[successor][var].variables.add(newname)


def _rename_node(state: SDFGState, node: nd.AccessNode, newname: str):
    # helper function that traverses memlet trees of all incoming and outgoing
    # edges of an accessnode and renames it to newname
    old_name = node.data
    node.data = newname
    for iedge in state.in_edges(node):
        for edge in state.memlet_tree(iedge):
            if edge.data.data == old_name:
                edge.data.data = newname
    for oedge in state.out_edges(node):
        for edge in state.memlet_tree(oedge):
            if edge.data.data == old_name:
                edge.data.data = newname


def _dominator_tree_DFS_order(
        sdfg: SDFG,
        immediate_dominators: Dict[SDFGState, SDFGState]
) -> List[SDFGState]:
    # helper function that returns the dominator tree of the SDFG in DFS order

    # dictionary mapping each state to the set of states dominated by that state
    immediate_dominated = defaultdict(None)
    for node, dominator_node in immediate_dominators.items():
        if dominator_node not in immediate_dominated:
            immediate_dominated[dominator_node] = set()
        immediate_dominated[dominator_node].add(node)

    graph_dict = immediate_dominated
    stack = []
    visited = []
    stack.append(sdfg.start_state)
    while stack:
        current_state: SDFGState = stack.pop()
        if current_state in visited:
            continue
        visited.append(current_state)
        if not current_state in graph_dict:
            continue
        children = graph_dict[current_state]
        for child in children:
            if child not in visited:
                stack.append(child)
    return visited


def _conditional_dfs(graph: SDFG, start: SDFG = None, condition=None):
    successors = graph.successors
    visited = set()
    node = start
    stack = [node]
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            if condition(node):
                stack.extend(iter(successors(node)))
    return visited


def _find_defining_states(
        sdfg: SDFG,
        array_variable_names: List[str],
        phi_nodes: Dict[SDFGState, Dict[str, _PhiNode]],
        access_nodes:  Dict[str, Dict[SDFGState, Tuple[Set[nd.AccessNode], Set[nd.AccessNode]]]],
        write_approximation: dict[Edge, Memlet]
) -> Dict[str, set[SDFGState]]:
    def_states: Dict[str, Set[SDFGState]] = {}

    for var in array_variable_names:
        # iterate over access nodes to the array in the current state and check if it
        # fully overwrites the array with the write underapproximation
        desc = sdfg.arrays[var]
        array_set = subsets.Range.from_array(desc)
        defining_states = set()
        for state in sdfg.states():
            # loopheaders that have phi nodes are also defining states
            if state in phi_nodes.keys():
                if var in phi_nodes[state].keys():
                    defining_states.add(state)
            # check if there is a write to the descriptor in the current state
            write_nodes = access_nodes[var][state][1]
            if len(write_nodes) == 0:
                continue
            for node in write_nodes:
                # if any of the edges fully overwrites the array add the state to
                # the defining states
                if any(write_approximation[edge].subset.covers_precise(array_set)
                        for edge in state.in_edges(node)):
                    defining_states.add(state)
                    break
        def_states[var] = defining_states

    return def_states


def _insert_phi_nodes_loopheaders(
        sdfg: SDFG,
        array_variable_names: List[str],
        loop_write_approximation: Dict[SDFGState, Dict[str, Memlet]],
        phi_nodes: Dict[SDFGState, Dict[str, _PhiNode]]
) -> Dict[str, Set[SDFGState]]:
    def_states_phi: Dict[str, Set[SDFGState]] = {}
    for loopheader, write_dict in loop_write_approximation.items():
        if loopheader not in sdfg.states():
            continue
        for var, memlet in write_dict.items():
            if loopheader in phi_nodes.keys() and var in phi_nodes[loopheader].keys():
                continue
            if var not in array_variable_names:
                continue
            if memlet.subset.covers_precise(subsets.Range.from_array(sdfg.arrays[var])):
                phi_nodes[loopheader][var] = _PhiNode(var, set())
                if var not in def_states_phi:
                    def_states_phi[var] = set()
                def_states_phi[var].add(loopheader)
    return def_states_phi


def _insert_phi_nodes_regular(
        sdfg: SDFG,
        def_states: Dict[str, Set[SDFGState]],
        phi_nodes: Dict[SDFGState, Dict[str, _PhiNode]],
        def_states_phi: Dict[str, Set[SDFGState]]):
    dominance_frontiers = nx.dominance.dominance_frontiers(
        sdfg.nx, sdfg.start_state)
    for var, defining_states in def_states.items():
        phi_states = set()
        defining_states = defining_states.copy()
        # array of states that define/fully overwrite the array
        while defining_states:
            current_state = next(iter(defining_states))
            dominance_frontier = dominance_frontiers[current_state]

            for frontier_state in dominance_frontier:
                # check if this state was already handled
                if frontier_state in phi_states:
                    continue
                phi_nodes[frontier_state][var] = _PhiNode(var, set())
                phi_states.add(frontier_state)
                if frontier_state not in defining_states:
                    defining_states.add(frontier_state)

            defining_states.remove(current_state)

        for state, phi_dict in phi_nodes.items():
            if var not in phi_dict.keys():
                continue
            if var not in def_states_phi:
                def_states_phi[var] = set()
            def_states_phi[var].add(state)
