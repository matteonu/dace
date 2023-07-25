import pytest

import dace
from dace.transformation.pass_pipeline import Pipeline
from dace.transformation.passes.scalar_fission import ScalarFission
from dace.sdfg.propagation_underapproximation import UnderapproximateWrites
from dace.subsets import Subset, Range





def test_simple_loop_overwrite():
    "simple loop that overwrites a one-dimensional array"
    N = dace.symbol("N")

    sdfg = dace.SDFG("simple_loop")
    iter_var1 = sdfg.add_symbol("i", dace.int32)
    sdfg.add_array("A", [N], dace.int64)
    init = sdfg.add_state("init")
    end = sdfg.add_state("end")
    loop_body = sdfg.add_state("loop_body")

    a0 = loop_body.add_access("A")
    loop_tasklet = loop_body.add_tasklet("overwrite", {} ,{"a"}, "a = 0" )
    loop_body.add_edge(loop_tasklet, "a", a0, None, dace.Memlet("A[i]"))


    _,guard,_ =sdfg.add_loop(init, loop_body, end, "i", "0", "i < N", "i + 1")

    pipeline = Pipeline([UnderapproximateWrites()])
    result = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]["loop_approximation"]


    assert(result[guard]["A"].subset.__str__() == Range.from_array(sdfg.arrays["A"]).__str__())

def test_loop_2D_overwrite():

    """two nested loops that overwrite a two-dimensional array"""
    N = dace.symbol("N")
    M = dace.symbol("M")

    sdfg = dace.SDFG("loop_2D_overwrite")
    iter_var1 = sdfg.add_symbol("i", dace.int32)
    iter_var2 = sdfg.add_symbol("j", dace.int32)
    sdfg.add_array("A", [M,N], dace.int64)
    init = sdfg.add_state("init")
    end = sdfg.add_state("end")
    loop_body = sdfg.add_state("loop_body")

    a0 = loop_body.add_access("A")
    loop_tasklet = loop_body.add_tasklet("overwrite", {} ,{"a"}, "a = 0" )
    loop_body.add_edge(loop_tasklet, "a", a0, None, dace.Memlet("A[j,i]"))

    loop_before_1 = sdfg.add_state("loop_before_1")
    loop_after_1 = sdfg.add_state("loop_after_1")
    _,guard2,_ =sdfg.add_loop(loop_before_1, loop_body, loop_after_1, "i", "0", "i < N", "i + 1")
    _,guard1,_ =sdfg.add_loop(init, loop_before_1, end, "j", "0", "j < M", "j + 1", loop_after_1)
    

    pipeline = Pipeline([UnderapproximateWrites()])
    result = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]["loop_approximation"]
    assert(result[guard1]["A"].subset.__str__() == Range.from_array(sdfg.arrays["A"]).__str__())
    assert(result[guard2]["A"].subset.__str__() == "j, 0:N")


def test_loop_2D_no_overwrite_0():

    """three nested loops that overwrite two dimensional array. 
    the innermost loop is surrounded by loop that doesn't iterate over array range.
    Therefore we don't want the full write in the approximation for the outermost loop,
    since the second innermost loop could just not execute"""
    N = dace.symbol("N")
    M = dace.symbol("M")
    K = dace.symbol("K")

    sdfg = dace.SDFG("loop_2D_no_overwrite")
    iter_var1 = sdfg.add_symbol("i", dace.int32)
    iter_var2 = sdfg.add_symbol("j", dace.int32)
    iter_var2 = sdfg.add_symbol("k", dace.int32)
    sdfg.add_array("A", [M,N], dace.int64)
    init = sdfg.add_state("init")
    end = sdfg.add_state("end")
    loop_body = sdfg.add_state("loop_body")

    a0 = loop_body.add_access("A")
    loop_tasklet = loop_body.add_tasklet("overwrite", {} ,{"a"}, "a = 0" )
    loop_body.add_edge(loop_tasklet, "a", a0, None, dace.Memlet("A[j,i]"))

    loop_before_1 = sdfg.add_state("loop_before_1")
    loop_after_1 = sdfg.add_state("loop_after_1")
    loop_before_2 = sdfg.add_state("loop_before_2")
    loop_after_2 = sdfg.add_state("loop_after_2")


    _,guard3,_ =sdfg.add_loop(loop_before_1, loop_body, loop_after_1, "i", "0", "i < N", "i + 1")   #inner-most loop
    _,guard2,_ = sdfg.add_loop(loop_before_2, loop_before_1, loop_after_2, "k", "0", "k < K", "k + 1", loop_after_1)    # second-inner-most loop
    _,guard1,_ =sdfg.add_loop(init, loop_before_2, end, "j", "0", "j < M", "j + 1", loop_after_2)   # outer-most loop
    
    pipeline = Pipeline([UnderapproximateWrites()])
    result = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]["loop_approximation"]
    assert("A" not in result[guard1].keys())
    assert("A" not in result[guard2].keys())
    assert(result[guard3]["A"].subset.__str__() == "j, 0:N")

def test_2_loops_overwrite():

    """2 loops one after another overwriting an array"""

    N = dace.symbol("N")

    sdfg = dace.SDFG("two_loops_overwrite")
    iter_var1 = sdfg.add_symbol("i", dace.int32)
    sdfg.add_array("A", [N], dace.int64)
    init = sdfg.add_state("init")
    end = sdfg.add_state("end")
    loop_body_1 = sdfg.add_state("loop_body_1")
    loop_body_2 = sdfg.add_state("loop_body_2")


    a0 = loop_body_1.add_access("A")
    loop_tasklet_1 = loop_body_1.add_tasklet("overwrite", {} ,{"a"}, "a = 0" )
    loop_body_1.add_edge(loop_tasklet_1, "a", a0, None, dace.Memlet("A[i]"))

    a1 = loop_body_2.add_access("A")
    loop_tasklet_2 = loop_body_2.add_tasklet("overwrite", {} ,{"a"}, "a = 0" )
    loop_body_2.add_edge(loop_tasklet_2, "a", a1, None, dace.Memlet("A[i]"))

    _,guard_1,after_state =sdfg.add_loop(init, loop_body_1, None, "i", "0", "i < N", "i + 1")

    _,guard_2,_ = sdfg.add_loop(after_state, loop_body_2, end, "i", "0", "i < N", "i + 1")
    pipeline = Pipeline([UnderapproximateWrites()])
    result = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]["loop_approximation"]

    assert(result[guard_1]["A"].subset.__str__() == Range.from_array(sdfg.arrays["A"]).__str__())
    assert(result[guard_2]["A"].subset.__str__() == Range.from_array(sdfg.arrays["A"]).__str__())

def test_loop_2D_overwrite_propagation_gap():

    """three nested loops that overwrite two dimensional array. 
    the innermost loop is surrounded by a loop that doesn't iterate over array range but over a non-empty constant range.
    Therefore the loop nest as a whole overwrites the array"""
    N = dace.symbol("N")
    M = dace.symbol("M")
    K = dace.symbol("K")

    sdfg = dace.SDFG("loop_2D_no_overwrite")
    iter_var1 = sdfg.add_symbol("i", dace.int32)
    iter_var2 = sdfg.add_symbol("j", dace.int32)
    iter_var2 = sdfg.add_symbol("k", dace.int32)
    sdfg.add_array("A", [M,N], dace.int64)
    init = sdfg.add_state("init")
    end = sdfg.add_state("end")
    loop_body = sdfg.add_state("loop_body")

    a0 = loop_body.add_access("A")
    loop_tasklet = loop_body.add_tasklet("overwrite", {} ,{"a"}, "a = 0" )
    loop_body.add_edge(loop_tasklet, "a", a0, None, dace.Memlet("A[j,i]"))

    loop_before_1 = sdfg.add_state("loop_before_1")
    loop_after_1 = sdfg.add_state("loop_after_1")
    loop_before_2 = sdfg.add_state("loop_before_2")
    loop_after_2 = sdfg.add_state("loop_after_2")


    _,guard3,_ =sdfg.add_loop(loop_before_1, loop_body, loop_after_1, "i", "0", "i < N", "i + 1")
    _,guard2,_ = sdfg.add_loop(loop_before_2, loop_before_1, loop_after_2, "k", "0", "k < 10", "k + 1", loop_after_1)
    _,guard1,_ =sdfg.add_loop(init, loop_before_2, end, "j", "0", "j < M", "j + 1", loop_after_2)
    

    pipeline = Pipeline([UnderapproximateWrites()])
    result = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]["loop_approximation"]
    assert(result[guard1]["A"].subset.__str__() == Range.from_array(sdfg.arrays["A"]).__str__())
    assert(result[guard2]["A"].subset.__str__() == "j, 0:N")
    assert(result[guard3]["A"].subset.__str__() == "j, 0:N")

def test_loop_2D_no_overwrite_1():

    """three nested loops that write to two dimensional array. 
    the subscript expression is a multiplication of two indices -> return empty subset"""
    N = dace.symbol("N")
    M = dace.symbol("M")
    K = dace.symbol("K")

    sdfg = dace.SDFG("loop_2D_no_overwrite")
    iter_var1 = sdfg.add_symbol("i", dace.int32)
    iter_var2 = sdfg.add_symbol("j", dace.int32)
    iter_var2 = sdfg.add_symbol("k", dace.int32)
    sdfg.add_array("A", [N,N], dace.int64)
    init = sdfg.add_state("init")
    end = sdfg.add_state("end")
    loop_body = sdfg.add_state("loop_body")

    a0 = loop_body.add_access("A")
    loop_tasklet = loop_body.add_tasklet("overwrite", {} ,{"a"}, "a = 0" )
    loop_body.add_edge(loop_tasklet, "a", a0, None, dace.Memlet("A[i,3*j]"))

    loop_before_1 = sdfg.add_state("loop_before_1")
    loop_after_1 = sdfg.add_state("loop_after_1")
    loop_before_2 = sdfg.add_state("loop_before_2")
    loop_after_2 = sdfg.add_state("loop_after_2")


    _,guard3,_ =sdfg.add_loop(loop_before_1, loop_body, loop_after_1, "i", "0", "i < N", "i + 1")
    _,guard2,_ = sdfg.add_loop(loop_before_2, loop_before_1, loop_after_2, "k", "0", "k < 10", "k + 1", loop_after_1)
    _,guard1,_ =sdfg.add_loop(init, loop_before_2, end, "j", "0", "j < M", "j + 1", loop_after_2)
    

    pipeline = Pipeline([UnderapproximateWrites()])
    result = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]["loop_approximation"]

    assert(result[guard1]["A"].subset.__str__() == Range.from_string("0:N, 0:3*M - 2:3").__str__())
    assert(result[guard2]["A"].subset.__str__() == "0:N, 3*j")
    assert(result[guard3]["A"].subset.__str__() == "0:N, 3*j")

def test_loop_2D_no_overwrite_2():

    """three nested loops that write to two dimensional array. 
    the innermost loop is surrounded by a loop that iterates over an empty range.
    Therefore the loop nest as a whole does not overwrite the array"""
    N = dace.symbol("N")
    M = dace.symbol("M")
    K = dace.symbol("K")

    sdfg = dace.SDFG("loop_2D_no_overwrite")
    iter_var1 = sdfg.add_symbol("i", dace.int32)
    iter_var2 = sdfg.add_symbol("j", dace.int32)
    iter_var2 = sdfg.add_symbol("k", dace.int32)
    sdfg.add_array("A", [M,N], dace.int64)
    init = sdfg.add_state("init")
    end = sdfg.add_state("end")
    loop_body = sdfg.add_state("loop_body")

    a0 = loop_body.add_access("A")
    loop_tasklet = loop_body.add_tasklet("overwrite", {} ,{"a"}, "a = 0" )
    loop_body.add_edge(loop_tasklet, "a", a0, None, dace.Memlet("A[j,i]"))

    loop_before_1 = sdfg.add_state("loop_before_1")
    loop_after_1 = sdfg.add_state("loop_after_1")
    loop_before_2 = sdfg.add_state("loop_before_2")
    loop_after_2 = sdfg.add_state("loop_after_2")


    _,guard3,_ =sdfg.add_loop(loop_before_1, loop_body, loop_after_1, "i", "0", "i < N", "i + 1")
    _,guard2,_ = sdfg.add_loop(loop_before_2, loop_before_1, loop_after_2, "k", "0", "k < 0", "k + 1", loop_after_1)
    _,guard1,_ =sdfg.add_loop(init, loop_before_2, end, "j", "0", "j < M", "j + 1", loop_after_2)
    

    pipeline = Pipeline([UnderapproximateWrites()])
    result = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]["loop_approximation"]
    assert(guard1 not in result.keys() or "A" not in result[guard1].keys())
    assert(guard2 not in result.keys() or "A" not in result[guard2].keys())
    assert(guard3 not in result.keys() or "A" not in result[guard3].keys())

def test_loop_2D_branch():

    """loop nested in another loop. nested loop is in a branch and overwrites the array.
        The propagation should return an empty subset for the outermost loop
        and the full subset for the loop in the branch"""
    N = dace.symbol("N")


    sdfg = dace.SDFG("loop_2D_branch")
    iter_var1 = sdfg.add_symbol("i", dace.int32)
    iter_var2 = sdfg.add_symbol("j", dace.int32)
    iter_var2 = sdfg.add_symbol("k", dace.int32)
    sdfg.add_array("A", [N], dace.int64)
    init = sdfg.add_state("init")
    end = sdfg.add_state("end")
    loop_body = sdfg.add_state("loop_body")

    a0 = loop_body.add_access("A")
    loop_tasklet = loop_body.add_tasklet("overwrite", {} ,{"a"}, "a = 0" )
    loop_body.add_edge(loop_tasklet, "a", a0, None, dace.Memlet("A[k]"))

    if_guard = sdfg.add_state("if_guard")
    if_merge = sdfg.add_state("if_merge")
    loop_before_2 = sdfg.add_state("loop_before_2")
    loop_after_2 = sdfg.add_state("loop_after_2")



    _,guard2,_ = sdfg.add_loop(loop_before_2, loop_body, loop_after_2, "k", "0", "k < N", "k + 1")
    sdfg.add_edge(if_guard, loop_before_2, dace.InterstateEdge(condition= "j % 2 == 0"))
    sdfg.add_edge(loop_after_2, if_merge, dace.InterstateEdge())
    sdfg.add_edge(if_guard, if_merge, dace.InterstateEdge(condition= "j % 2 == 1"))
    _,guard1,_ =sdfg.add_loop(init, if_guard, end, "j", "0", "j < M", "j + 1", if_merge)
    
    pipeline = Pipeline([UnderapproximateWrites()])
    result = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]["loop_approximation"]
    assert(guard1 not in result.keys() or "A" not in result[guard1].keys())
    assert(guard2  in result.keys() and "A"  in result[guard2].keys())


def test_loop_nested_sdfg():

    M = dace.symbol("M")
    N = dace.symbol("N")

    @dace.program
    def nested_loop(A: dace.float64[M, N]):
        for i in dace.map[0:M]:
            for j in range(N):
                A[i + 1,j * i] = 1

    sdfg = nested_loop.to_sdfg()
    pipeline = Pipeline([UnderapproximateWrites()])
    result = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]

def test_loop_in_nested_sdfg_simple():

    M = dace.symbol("M")
    N = dace.symbol("N")

    @dace.program
    def nested_loop(A: dace.float64[M, N]):
        for i in dace.map[0:M]:
            for j in range(N):
                A[i ,j] = 1

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


    assert(write_set.__str__() == "0:M, 0:N" )


def test_loop_break():
    """loop that has a break statement. So the analysis should not propagate memlets outside of this loop"""
    N = dace.symbol("N")

    sdfg = dace.SDFG("loop_2D_no_overwrite")
    sdfg.add_array("A", [N], dace.int64)
    init = sdfg.add_state("init", is_start_state=True)
    loop_body_0 = sdfg.add_state("loop_body_0")
    loop_body_1 = sdfg.add_state("loop_body_1")

    loop_after_1 = sdfg.add_state("loop_after_1")
    sdfg.add_edge(loop_body_0, loop_after_1, dace.InterstateEdge(condition = "i > 10"))
    sdfg.add_edge(loop_body_0, loop_body_1, dace.InterstateEdge(condition="not(i > 10)"))
    a0 = loop_body_1.add_access("A")
    loop_tasklet = loop_body_1.add_tasklet("overwrite", {} ,{"a"}, "a = 0" )
    loop_body_1.add_edge(loop_tasklet, "a", a0, None, dace.Memlet("A[i]"))



    _,guard3,_ =sdfg.add_loop(init, loop_body_0, loop_after_1, "i", "0", "i < N", "i + 1",loop_body_1)

    pipeline = Pipeline([UnderapproximateWrites()])
    results = pipeline.apply_pass(sdfg, {})[UnderapproximateWrites.__name__]
    result = results["loop_approximation"]
    assert(guard3 not in result.keys() or "A" not in result[guard3].keys())





if __name__ == '__main__':
    test_loop_in_nested_sdfg_simple()
    test_loop_2D_branch()
    test_loop_2D_no_overwrite_2()
    test_simple_loop_overwrite()
    test_loop_2D_overwrite()
    test_loop_2D_overwrite_propagation_gap()
    test_2_loops_overwrite()
    test_loop_2D_no_overwrite_0()
    test_loop_2D_no_overwrite_1()
    test_loop_nested_sdfg()
    test_loop_break()
