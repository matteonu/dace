import pytest

import dace
from dace.transformation.pass_pipeline import Pipeline
from dace.transformation.passes.scalar_fission import ScalarFission
from dace.sdfg.propagation_underapproximation import UnderapproximateWrites
from dace.subsets import Subset, Range



def test_2D_map_full_write():
    """2-dimensional map that fully overwrites 2-dimensional array --> propagate full range"""

    sdfg = dace.SDFG("twoD_map")
    M = dace.symbol("M")
    N = dace.symbol("N")
    sdfg.add_array("B", (M,N), dace.float64)
    map_state = sdfg.add_state("map")
    a1 = map_state.add_access('B')
    map_state.add_mapped_tasklet("overwrite_1", 
                map_ranges = {
                    '_i': f'0:N:1',
                    '_j': f'0:M:1'
                },
                inputs = {},
                code = f"b = 5",
                outputs= {
                    "b": dace.Memlet("B[_j,_i]")
                },
                output_nodes={"B": a1},
                external_edges=True
                )
    
    edge = map_state.in_edges(a1)[0]
    pipeline = Pipeline([UnderapproximateWrites()])
    results = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]
    result = results["approximation"]
    expected_subset = Range.from_string("0:M, 0:N")
    result_subset_list = result[edge].subset.subset_list
    result_subset = result_subset_list[0]
    assert(result_subset.__str__() == expected_subset.__str__())

def test_2D_map_added_indices_no_propagation():
    """2-dimensional array that writes to two-dimensional array with 
    subscript expression that adds two indices --> empty subset"""

    sdfg = dace.SDFG("twoD_map")
    M = dace.symbol("M")
    N = dace.symbol("N")
    sdfg.add_array("B", (M,N), dace.float64)
    map_state = sdfg.add_state("map")
    a1 = map_state.add_access('B')
    map_state.add_mapped_tasklet("overwrite_1", 
                map_ranges = {
                    '_i': f'0:N:1',
                    '_j': f'0:M:1'
                },
                inputs = {},
                code = f"b = 5",
                outputs= {
                    "b": dace.Memlet("B[_j,_i + _j]")
                },
                output_nodes={"B": a1},
                external_edges=True
                )
    
    edge = map_state.in_edges(a1)[0]
    pipeline = Pipeline([UnderapproximateWrites()])
    results = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]
    result = results["approximation"]
    expected_subset = Range.from_string("0:M, 0:N")
    assert (len(result[edge].subset.subset_list) == 0)

def test_2D_map_multiplied_indices_no_propagation():
    """2-dimensional array that writes to two-dimensional array with 
    subscript expression that multiplies two indices --> empty subset"""


    sdfg = dace.SDFG("twoD_map")
    M = dace.symbol("M")
    N = dace.symbol("N")
    sdfg.add_array("B", (M,N), dace.float64)
    map_state = sdfg.add_state("map")
    a1 = map_state.add_access('B')
    map_state.add_mapped_tasklet("overwrite_1", 
                map_ranges = {
                    '_i': f'0:N:1',
                    '_j': f'0:M:1'
                },
                inputs = {},
                code = f"b = 5",
                outputs= {
                    "b": dace.Memlet("B[_j,_i * _j]")
                },
                output_nodes={"B": a1},
                external_edges=True
                )
    
    edge = map_state.in_edges(a1)[0]
    pipeline = Pipeline([UnderapproximateWrites()])
    results = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]
    result = results["approximation"]
    expected_subset = Range.from_string("0:M, 0:N")
    assert (len(result[edge].subset.subset_list) == 0)

def test_1D_map_one_index_multiple_dims_no_propagation():
    """one dimensional map that has the same index 
    in two dimensions in a write-access --> no propagation"""

    sdfg = dace.SDFG("twoD_map")
    M = dace.symbol("M")
    N = dace.symbol("N")
    sdfg.add_array("B", (M,N), dace.float64)
    map_state = sdfg.add_state("map")
    a1 = map_state.add_access('B')
    map_state.add_mapped_tasklet("overwrite_1", 
                map_ranges = {
                    '_j': f'0:M:1'
                },
                inputs = {},
                code = f"b = 5",
                outputs= {
                    "b": dace.Memlet("B[_j, _j]")
                },
                output_nodes={"B": a1},
                external_edges=True
                )
    
    edge = map_state.in_edges(a1)[0]
    pipeline = Pipeline([UnderapproximateWrites()])
    results = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]
    result = results["approximation"]
    expected_subset = Range.from_string("0:M, 0:N")
    assert (len(result[edge].subset.subset_list) == 0)

def test_1D_map_one_index_squared_no_fission():
    """one dimensional map that multiplies the index 
    in the subscript expression --> no propagation"""

    sdfg = dace.SDFG("twoD_map")
    M = dace.symbol("M")
    N = dace.symbol("N")
    sdfg.add_array("B", (M,), dace.float64)
    map_state = sdfg.add_state("map")
    a1 = map_state.add_access('B')
    map_state.add_mapped_tasklet("overwrite_1", 
                map_ranges = {
                    '_j': f'0:M:1'
                },
                inputs = {},
                code = f"b = 5",
                outputs= {
                    "b": dace.Memlet("B[_j* _j]")
                },
                output_nodes={"B": a1},
                external_edges=True
                )
    
    edge = map_state.in_edges(a1)[0]
    pipeline = Pipeline([UnderapproximateWrites()])
    results = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]
    result = results["approximation"]
    expected_subset = Range.from_string("0:M, 0:N")
    assert (len(result[edge].subset.subset_list) == 0)


def test_map_tree_full_write():
    """two maps nested in map. both maps overwrite the whole first dimension of the array
    together with the outer map the whole array is overwritten"""

    sdfg = dace.SDFG("twoD_map")
    M = dace.symbol("M")
    N = dace.symbol("N")
    sdfg.add_array("B", (M,N), dace.float64)
    map_state = sdfg.add_state("map")
    a1 = map_state.add_access('B')
    map_entry, map_exit = map_state.add_map("outer_map", {"_i": f'0:N:1'})
    map_tasklet_0 = map_state.add_tasklet("map_tasklet_0", {}, {"b"}, "b = 1")
    map_tasklet_1 = map_state.add_tasklet("map_tasklet_1", {}, {"b"}, "b = 2")
    map_exit.add_in_connector("IN_B")
    map_exit.add_out_connector("OUT_B")

    
    inner_map_entry_0, inner_map_exit_0 = map_state.add_map("inner_map_0", {"_j": f'0:M:1'})
    inner_map_exit_0.add_in_connector("IN_B")
    inner_map_exit_0.add_out_connector("OUT_B")
    inner_map_entry_1, inner_map_exit_1 = map_state.add_map("inner_map_1", {"_j": f'0:M:1'})
    inner_map_exit_1.add_in_connector("IN_B")
    inner_map_exit_1.add_out_connector("OUT_B")
    
    map_state.add_edge(map_entry, None, inner_map_entry_0, None, dace.Memlet())
    map_state.add_edge(inner_map_entry_0, None, map_tasklet_0, None, dace.Memlet())
    map_state.add_edge(map_tasklet_0, "b", inner_map_exit_0, "IN_B", dace.Memlet("B[_j, _i]"))
    inner_edge_0 = map_state.add_edge(inner_map_exit_0, "OUT_B", map_exit, "IN_B", dace.Memlet(data = "B"))
    
    map_state.add_edge(map_entry, None, inner_map_entry_1, None, dace.Memlet())
    map_state.add_edge(inner_map_entry_1, None, map_tasklet_1, None, dace.Memlet())
    map_state.add_edge(map_tasklet_1, "b", inner_map_exit_1, "IN_B", dace.Memlet("B[_j, _i]"))
    inner_edge_1 = map_state.add_edge(inner_map_exit_1, "OUT_B", map_exit, "IN_B", dace.Memlet(data = "B"))

    outer_edge = map_state.add_edge(map_exit, "OUT_B", a1, None, dace.Memlet(data = "B"))

    pipeline = Pipeline([UnderapproximateWrites()])
    results = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]
    result = results["approximation"]


    expected_subset_outer_edge = Range.from_string("0:M, 0:N")
    expected_subset_inner_edge = Range.from_string("0:M, _i")
    result_inner_edge_0 = result[inner_edge_0].subset.subset_list[0]
    result_inner_edge_1 = result[inner_edge_1].subset.subset_list[0]
    result_outer_edge = result[outer_edge].subset.subset_list[0]

    assert(result_inner_edge_0.__str__() == expected_subset_inner_edge.__str__())
    assert(result_inner_edge_1.__str__() == expected_subset_inner_edge.__str__())
    assert(result_outer_edge.__str__() == expected_subset_outer_edge.__str__())

def test_map_tree_no_write_multiple_indices():
    """same as test_map_tree_full_write but now the accesses 
    in both nested maps add two indices --> no propagation"""

    sdfg = dace.SDFG("twoD_map")
    M = dace.symbol("M")
    N = dace.symbol("N")
    sdfg.add_array("B", (M,N), dace.float64)
    map_state = sdfg.add_state("map")
    a1 = map_state.add_access('B')
    map_entry, map_exit = map_state.add_map("outer_map", {"_i": f'0:N:1'})
    map_tasklet_0 = map_state.add_tasklet("map_tasklet_0", {}, {"b"}, "b = 1")
    map_tasklet_1 = map_state.add_tasklet("map_tasklet_1", {}, {"b"}, "b = 2")
    map_exit.add_in_connector("IN_B")
    map_exit.add_out_connector("OUT_B")

    
    inner_map_entry_0, inner_map_exit_0 = map_state.add_map("inner_map_0", {"_j": f'0:M:1'})
    inner_map_exit_0.add_in_connector("IN_B")
    inner_map_exit_0.add_out_connector("OUT_B")
    inner_map_entry_1, inner_map_exit_1 = map_state.add_map("inner_map_1", {"_j": f'0:M:1'})
    inner_map_exit_1.add_in_connector("IN_B")
    inner_map_exit_1.add_out_connector("OUT_B")
    
    map_state.add_edge(map_entry, None, inner_map_entry_0, None, dace.Memlet())
    map_state.add_edge(inner_map_entry_0, None, map_tasklet_0, None, dace.Memlet())
    map_state.add_edge(map_tasklet_0, "b", inner_map_exit_0, "IN_B", dace.Memlet("B[_j + _i, _i]"))
    inner_edge_0 = map_state.add_edge(inner_map_exit_0, "OUT_B", map_exit, "IN_B", dace.Memlet(data = "B"))
    
    map_state.add_edge(map_entry, None, inner_map_entry_1, None, dace.Memlet())
    map_state.add_edge(inner_map_entry_1, None, map_tasklet_1, None, dace.Memlet())
    map_state.add_edge(map_tasklet_1, "b", inner_map_exit_1, "IN_B", dace.Memlet("B[_j, _i + _j]"))
    inner_edge_1 = map_state.add_edge(inner_map_exit_1, "OUT_B", map_exit, "IN_B", dace.Memlet(data = "B"))

    outer_edge = map_state.add_edge(map_exit, "OUT_B", a1, None, dace.Memlet(data = "B"))

    pipeline = Pipeline([UnderapproximateWrites()])
    results = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]
    result = results["approximation"]
    # import pprint
    # pprint.pprint(result)
    expected_subset_outer_edge = Range.from_string("0:M, 0:N")
    expected_subset_inner_edge = Range.from_string("0:M, _i")
    result_inner_edge_0 = result[inner_edge_0].subset.subset_list
    result_inner_edge_1 = result[inner_edge_1].subset.subset_list
    result_outer_edge = result[outer_edge].subset.subset_list

    assert(len(result_inner_edge_0) == 0)
    assert(len(result_inner_edge_1) == 0)
    assert(len(result_outer_edge) == 0)

def test_map_tree_multiple_indices_per_dimension_full_write():
    """same as test_map_tree_full_write but one of the inner maps squares index _j --> full write"""

    sdfg = dace.SDFG("twoD_map")
    M = dace.symbol("M")
    N = dace.symbol("N")
    sdfg.add_array("B", (M,N), dace.float64)
    map_state = sdfg.add_state("map")
    a1 = map_state.add_access('B')
    map_entry, map_exit = map_state.add_map("outer_map", {"_i": f'0:N:1'})
    map_tasklet_0 = map_state.add_tasklet("map_tasklet_0", {}, {"b"}, "b = 1")
    map_tasklet_1 = map_state.add_tasklet("map_tasklet_1", {}, {"b"}, "b = 2")
    map_exit.add_in_connector("IN_B")
    map_exit.add_out_connector("OUT_B")

    
    inner_map_entry_0, inner_map_exit_0 = map_state.add_map("inner_map_0", {"_j": f'0:M:1'})
    inner_map_exit_0.add_in_connector("IN_B")
    inner_map_exit_0.add_out_connector("OUT_B")
    inner_map_entry_1, inner_map_exit_1 = map_state.add_map("inner_map_1", {"_j": f'0:M:1'})
    inner_map_exit_1.add_in_connector("IN_B")
    inner_map_exit_1.add_out_connector("OUT_B")
    
    map_state.add_edge(map_entry, None, inner_map_entry_0, None, dace.Memlet())
    map_state.add_edge(inner_map_entry_0, None, map_tasklet_0, None, dace.Memlet())
    map_state.add_edge(map_tasklet_0, "b", inner_map_exit_0, "IN_B", dace.Memlet("B[_j * _j, _i ]"))
    inner_edge_0 = map_state.add_edge(inner_map_exit_0, "OUT_B", map_exit, "IN_B", dace.Memlet(data = "B"))
    
    map_state.add_edge(map_entry, None, inner_map_entry_1, None, dace.Memlet())
    map_state.add_edge(inner_map_entry_1, None, map_tasklet_1, None, dace.Memlet())
    map_state.add_edge(map_tasklet_1, "b", inner_map_exit_1, "IN_B", dace.Memlet("B[_j, _i]"))
    inner_edge_1 = map_state.add_edge(inner_map_exit_1, "OUT_B", map_exit, "IN_B", dace.Memlet(data = "B"))

    outer_edge = map_state.add_edge(map_exit, "OUT_B", a1, None, dace.Memlet(data = "B"))

    pipeline = Pipeline([UnderapproximateWrites()])
    results = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]
    result = results["approximation"]
    # import pprint
    # pprint.pprint(result)
    expected_subset_outer_edge = Range.from_string("0:M, 0:N")
    expected_subset_inner_edge_1 = Range.from_string("0:M, _i")

    result_inner_edge_1 = result[inner_edge_1].subset.subset_list[0]
    result_outer_edge = result[outer_edge].subset.subset_list[0]

    assert(len(result[inner_edge_0].subset.subset_list) == 0)
    assert(result_inner_edge_1.__str__() == expected_subset_inner_edge_1.__str__())
    assert(result_outer_edge.__str__() == expected_subset_outer_edge.__str__())



def test_loop_in_map_no_write():
    """loop in a map and indices are multiplied in subscript expression -> no propagation"""
    M = dace.symbol("M")
    N = dace.symbol("N")
    @dace.program
    def loop(A: dace.float64[N,M]):
        for i in dace.map[0:N]:
            for j in range(M):
                A[i,j * i] = 0

    sdfg = loop.to_sdfg()
    pipeline = Pipeline([UnderapproximateWrites()])
    result = pipeline.apply_pass(sdfg, {})
    sdfg.all_nodes_recursive()
    nsdfgs = set()
    nsdfg = sdfg.sdfg_list[1].parent_nsdfg_node


    map_state = sdfg.states()[0]
    edge = map_state.out_edges(nsdfg)[0]
    pipeline = Pipeline([UnderapproximateWrites()])
    results = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]
    result = results["approximation"]
    expected_subset = Range.from_string("0:M, 0:N")
    assert (len(result[edge].subset.subset_list) == 0)

def test_loop_in_map_full_write():
    """loop in map and both together fully overwrite the array"""

    M = dace.symbol("M")
    N = dace.symbol("N")
    @dace.program
    def loop(A: dace.float64[N,M]):
        for i in dace.map[0:N]:
            for j in range(M):
                A[i,j ] = 0

    sdfg = loop.to_sdfg()
    pipeline = Pipeline([UnderapproximateWrites()])
    result = pipeline.apply_pass(sdfg, {})
    sdfg.all_nodes_recursive()
    nsdfgs = set()
    nsdfg = sdfg.sdfg_list[1].parent_nsdfg_node

    map_state = sdfg.states()[0]
    edge = map_state.in_edges(map_state.data_nodes()[0])[0]
    pipeline = Pipeline([UnderapproximateWrites()])
    results = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]
    result = results["approximation"]
    expected_subset = Range.from_string("0:N, 0:M")
    assert (result[edge].subset.subset_list[0].__str__() == expected_subset.__str__())

def test_map_in_loop_full_write():
    """map in loop, together they overwrite a two dimensional array"""
    M = dace.symbol("M")
    N = dace.symbol("N")

    sdfg = dace.SDFG("nested")
    sdfg.add_array("B", (N,M), dace.float64)
    init = sdfg.add_state("init")
    guard = sdfg.add_state("guard")
    body = sdfg.add_state("body")
    end = sdfg.add_state("end")
    sdfg.add_edge(init, guard, dace.InterstateEdge(assignments={"j":"0"}))
    sdfg.add_edge(guard, body, dace.InterstateEdge(condition="j < N"))
    sdfg.add_edge(guard, end, dace.InterstateEdge(condition="not(j < N)"))
    sdfg.add_edge(body, guard, dace.InterstateEdge(assignments={"j":"j + 1"}))

    a1 = body.add_access("B")
    body.add_mapped_tasklet("overwrite_1", 
                map_ranges = {
                    'i': f'0:M:1'
                },
                inputs = {},
                code = f"b = 5",
                outputs= {
                    "b": dace.Memlet("B[j, i]")
                },
                output_nodes={"B": a1},
                external_edges=True
                )
    

    

    pipeline = Pipeline([UnderapproximateWrites()])
    results = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]
    result = results["loop_approximation"]
    expected_subset = Range.from_string("0:N, 0:M")
    assert (result[guard]["B"].subset.subset_list[0].__str__() == expected_subset.__str__())

def test_map_in_loop_no_write_0():
    """map in loop. Subscript expression of array access in loop multiplies two indicies --> no propagation"""

    M = dace.symbol("M")
    N = dace.symbol("N")

    sdfg = dace.SDFG("nested")
    sdfg.add_array("B", (N,M), dace.float64)
    init = sdfg.add_state("init")
    guard = sdfg.add_state("guard")
    body = sdfg.add_state("body")
    end = sdfg.add_state("end")
    sdfg.add_edge(init, guard, dace.InterstateEdge(assignments={"j":"0"}))
    sdfg.add_edge(guard, body, dace.InterstateEdge(condition="j < N"))
    sdfg.add_edge(guard, end, dace.InterstateEdge(condition="not(j < N)"))
    sdfg.add_edge(body, guard, dace.InterstateEdge(assignments={"j":"j + 1"}))

    a1 = body.add_access("B")
    body.add_mapped_tasklet("overwrite_1", 
                map_ranges = {
                    'i': f'0:M:1'
                },
                inputs = {},
                code = f"b = 5",
                outputs= {
                    "b": dace.Memlet("B[j * i, i]")
                },
                output_nodes={"B": a1},
                external_edges=True
                )
    

    pipeline = Pipeline([UnderapproximateWrites()])
    results = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]
    result = results["loop_approximation"]
    import pprint
    pprint.pprint(result)
    expected_subset = Range.from_string("0:N, 0:M")
    assert (guard not in result.keys() or len(result[guard]) == 0)

def test_map_in_loop_no_write_1():
    """same as test_map_in_loop_no_write_0 but for other dimension"""


    M = dace.symbol("M")
    N = dace.symbol("N")

    sdfg = dace.SDFG("nested")
    sdfg.add_array("B", (N,M), dace.float64)
    init = sdfg.add_state("init")
    guard = sdfg.add_state("guard")
    body = sdfg.add_state("body")
    end = sdfg.add_state("end")
    sdfg.add_edge(init, guard, dace.InterstateEdge(assignments={"j":"0"}))
    sdfg.add_edge(guard, body, dace.InterstateEdge(condition="j < N"))
    sdfg.add_edge(guard, end, dace.InterstateEdge(condition="not(j < N)"))
    sdfg.add_edge(body, guard, dace.InterstateEdge(assignments={"j":"j + 1"}))

    a1 = body.add_access("B")
    body.add_mapped_tasklet("overwrite_1", 
                map_ranges = {
                    'i': f'0:M:1'
                },
                inputs = {},
                code = f"b = 5",
                outputs= {
                    "b": dace.Memlet("B[j, i * j]")
                },
                output_nodes={"B": a1},
                external_edges=True
                )
    

    pipeline = Pipeline([UnderapproximateWrites()])
    results = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]
    result = results["loop_approximation"]
    import pprint
    pprint.pprint(result)
    expected_subset = Range.from_string("0:N, 0:M")
    assert (guard not in result.keys() or len(result[guard]) == 0)


def test_nested_sdfg_in_map_full_write():

    M = dace.symbol("M")
    N = dace.symbol("N")

    @dace.program
    def nested_loop(A: dace.float64[M, N]):
        for i in dace.map[0:M]:
            for j in dace.map[0:N]:
                if A[0]:
                    A[i ,j] = 1
                else:
                    A[i,j] = 2
                A[i,j] = A[i,j] * A[i,j]

    sdfg = nested_loop.to_sdfg()
    sdfg.view()
    pipeline = Pipeline([UnderapproximateWrites()])
    result = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]
    
    write_approx = result["approximation"]

    # find write set
    accessnode = None
    for node,_ in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.AccessNode):
            if node.data == "A":
                accessnode = node
    for edge, memlet in write_approx.items():
        if edge.dst is accessnode:
            write_set = memlet.subset

    assert(write_set.__str__() == "0:M, 0:N" )

def test_loop_in_nested_sdfg_in_map_partial_write():

    M = dace.symbol("M")
    N = dace.symbol("N")

    @dace.program
    def nested_loop(A: dace.float64[M, N]):
        for i in dace.map[0:M]:
            for j in range(2,N,1):
                if A[0]:
                    A[i ,j] = 1
                else:
                    A[i,j] = 2
                A[i,j] = A[i,j] * A[i,j]

    sdfg = nested_loop.to_sdfg()
    pipeline = Pipeline([UnderapproximateWrites()])
    result = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]
    
    write_approx = result["approximation"]

    # find write set
    accessnode = None
    for node,_ in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.AccessNode):
            if node.data == "A":
                accessnode = node
    for edge, memlet in write_approx.items():
        if edge.dst is accessnode:
            write_set = memlet.subset

    assert(write_set.__str__() == "0:M, 0:N - 2" )

def test_nested_sdfg_in_map_full_write_1():

    M = dace.symbol("M")
    N = dace.symbol("N")

    @dace.program
    def nested_loop(A: dace.float64[M, N]):
        for i in dace.map[0:M]:
                if A[0]:
                    A[i, :] = 1
                else:
                    A[i, :] = 2
                A[i, :] = 0

    sdfg = nested_loop.to_sdfg()
    sdfg.view()
    pipeline = Pipeline([UnderapproximateWrites()])
    result = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]
    
    write_approx = result["approximation"]

    # find write set
    accessnode = None
    for node,_ in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.AccessNode):
            if node.data == "A":
                accessnode = node
    for edge, memlet in write_approx.items():
        if edge.dst is accessnode:
            write_set = memlet.subset

    assert(write_set.__str__() == "0:M, 0:N" )

def test_nested_sdfg_in_map_branches_no_write():

    M = dace.symbol("M")
    N = dace.symbol("N")

    @dace.program
    def nested_loop(A: dace.float64[M, N]):
        for i in dace.map[0:M]:
                if A[0]:
                    A[i, :] = 1
                else:
                    A[i, :] = 2


    sdfg = nested_loop.to_sdfg()
    sdfg.view()
    pipeline = Pipeline([UnderapproximateWrites()])
    result = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]
    
    write_approx = result["approximation"]

    # find write set
    accessnode = None
    for node,_ in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.AccessNode):
            if node.data == "A":
                accessnode = node
    for edge, memlet in write_approx.items():
        if edge.dst is accessnode:
            write_set = memlet.subset

    assert(not write_set.__str__() == "0:M, 0:N" )





if __name__ == '__main__':
    test_nested_sdfg_in_map_branches_no_write()
    test_nested_sdfg_in_map_full_write_1()
    test_loop_in_nested_sdfg_in_map_partial_write()
    test_nested_sdfg_in_map_full_write()
    test_map_in_loop_no_write_0()
    test_map_in_loop_no_write_1()
    test_map_in_loop_full_write()
    test_loop_in_map_no_write()
    test_loop_in_map_full_write()
    test_map_tree_full_write()
    test_2D_map_full_write()
    test_2D_map_added_indices_no_propagation()
    test_2D_map_multiplied_indices_no_propagation()
    test_1D_map_one_index_multiple_dims_no_propagation()
    test_1D_map_one_index_squared_no_fission()
    test_map_tree_multiple_indices_per_dimension_full_write()
    test_map_tree_no_write_multiple_indices()

