import dace
from dace.sdfg.propagation_underapproximation import UnderapproximateWrites
from dace.subsets import Range
import pytest



def constant_multiplicative_2D_test():
    N = dace.symbol("N")
    M = dace.symbol("M")
    A = dace.data.Array(dace.int64, (N,M))
    subset = Range.from_string("i,3*j")
    i_subset = Range.from_string("0:N:1")
    j_subset = Range.from_string("0:M:1")
    memlet = dace.Memlet(None, "A", subset)
    memlets = [memlet]
    propagated_memlet = UnderapproximateWrites().propagate_subset(memlets,A,["j"], j_subset, None, True)
    propagated_memlet = UnderapproximateWrites().propagate_subset([propagated_memlet],A,["i"], i_subset, None, True)

    propagated_subset = propagated_memlet.subset.subset_list[0]
    expected_subset = Range.from_string("0:N:1, 0:3*M - 2:3")
    propagated_string = propagated_subset.__str__()
    expected_string = expected_subset.__str__()
    assert(propagated_string == expected_string)
 
def affine_2D_test():
    N = dace.symbol("N")
    M = dace.symbol("M")
    A = dace.data.Array(dace.int64, (N,M))
    subset = Range.from_string("i,3 * j + 3")
    i_subset = Range.from_string("0:N:1")
    j_subset = Range.from_string("0:M:1")
    memlet = dace.Memlet(None, "A", subset)
    memlets = [memlet]
    propagated_memlet = UnderapproximateWrites().propagate_subset(memlets,A,["j"], j_subset, None, True)
    propagated_memlet = UnderapproximateWrites().propagate_subset([propagated_memlet],A,["i"], i_subset, None, True)

    propagated_subset = propagated_memlet.subset.subset_list[0]
    expected_subset = Range.from_string("0:N:1, 3 : 3 * M + 1 : 3")
    propagated_string = propagated_subset.__str__()
    expected_string = expected_subset.__str__()
    assert(propagated_string == expected_string)

def multiplied_variables_test():
    N = dace.symbol("N", dace.float64)
    M = dace.symbol("M", dtype= dace.float64)
    A = dace.data.Array(dace.int64, (M,))
    subset = Range.from_string("i * j")
    i_subset = Range.from_string("0:N:1")
    j_subset = Range.from_string("0:M:1")
    memlet = dace.Memlet(None, "A", subset)
    memlets = [memlet]
    propagated_memlet = UnderapproximateWrites().propagate_subset(memlets,A,["j"], j_subset, None, True)
    propagated_memlet = UnderapproximateWrites().propagate_subset([propagated_memlet],A,["i"], i_subset, None, True)
    assert(not propagated_memlet.subset.subset_list)
    #assert that the propagated memlet is empty


def one_variable_in_2dimensions_test():
    N = dace.symbol("N")
    M = dace.symbol("M")
    A = dace.data.Array(dace.int64, (N,M))
    subset = Range.from_string("i, i")
    i_subset = Range.from_string("0:N:1")
    j_subset = Range.from_string("0:M:1")
    memlet = dace.Memlet(None, "A", subset)
    memlets = [memlet]
    propagated_memlet = UnderapproximateWrites().propagate_subset(memlets,A,["j"], j_subset, None, True)
    propagated_memlet = UnderapproximateWrites().propagate_subset([propagated_memlet],A,["i"], i_subset, None, True)
    print(propagated_memlet.subset)
    assert(not propagated_memlet.subset.subset_list)

def negative_step_test():
    N = dace.symbol("N")
    M = dace.symbol("M")
    A = dace.data.Array(dace.int64, (N,M))
    subset = Range.from_string("i, j")
    i_subset = Range.from_string("0:N:1")
    j_subset = Range.from_string("M:0:-1")
    memlet = dace.Memlet(None, "A", subset)
    memlets = [memlet]
    propagated_memlet = UnderapproximateWrites().propagate_subset(memlets,A,["j"], j_subset, None, True)
    propagated_memlet = UnderapproximateWrites().propagate_subset([propagated_memlet],A,["i"], i_subset, None, True)
    print(propagated_memlet.subset)
    propagated_subset = propagated_memlet.subset.subset_list[0]
    expected_subset = Range.from_string("0:N:1,0:M:1")

    propagated_string = propagated_subset.__str__()
    expected_string = expected_subset.__str__()
    assert(propagated_string == expected_string)
    assert(not propagated_memlet.subset.subset_list)

def step_not_one_test():
    N = dace.symbol("N")
    M = dace.symbol("M")
    A = dace.data.Array(dace.int64, (N,M))
    subset = Range.from_string("i")
    i_subset = Range.from_string("0:N:3")
    memlet = dace.Memlet(None, "A", subset)
    memlets = [memlet]
    propagated_memlet = UnderapproximateWrites().propagate_subset(memlets,A,["i"], i_subset, None, True)
    propagated_subset = propagated_memlet.subset.subset_list[0]
    expected_subset = Range.from_string("0:N:3")

    propagated_string = propagated_subset.__str__()
    expected_string = expected_subset.__str__()
    assert(propagated_string == expected_string)


def tiling_test():
    # test if tiling works. 
    N = dace.symbol("N")
    M = dace.symbol("M")
    A = dace.data.Array(dace.int64, (N,M))
    subset = Range.from_string("i, i")
    i_subset = Range.from_string("0:N:1")
    j_subset = Range.from_string("0:M:1")
    memlet = dace.Memlet(None, "A", subset)
    memlets = [memlet]
    propagated_memlet = UnderapproximateWrites().propagate_subset(memlets,A,["j"], j_subset, None, True)
    propagated_memlet = UnderapproximateWrites().propagate_subset([propagated_memlet],A,["i"], i_subset, None, True)
    # assert that memlet is empty




if __name__ == '__main__':
    step_not_one_test()
    one_variable_in_2dimensions_test()
    affine_2D_test()
    constant_multiplicative_2D_test()
    multiplied_variables_test()
    negative_step_test()