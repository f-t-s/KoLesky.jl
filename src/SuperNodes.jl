import Base.size

# In this file we introduce the data types for super nodes
abstract type AbstractSuperNode end

# A supernode that contains indices to measurements
struct IndexSuperNode{Ti}
    column_indices::Vector{Ti}
    row_indices::Vector{Ti}
end

function column_indices(node::IndexSuperNode) return node.column_indices end
function row_indices(node::IndexSuperNode) return node.row_indices end

function size(in::IndexSuperNode)
    return (length(in.column_indices) , length(in.row_indices))
end

function size(in::IndexSuperNode, dim)
    return (length(in.column_indices) , length(in.row_indices))[dim]
end


abstract type AbstractSupernodalAssignment end

# A supernodal assigment given in terms of a 
struct IndirectSupernodalAssignment{Ti<:Integer, Tm<:AbstractMeasurement}
    # A vector containing the index supernodes
    supernodes::Vector{IndexSuperNode{Ti}}
    # A vector containing the measurements
    measurements::Vector{Tm}
end

# presently buggy
# function istriu(node::IndexSuperNode)
#     return minimum(node.row_indices) >= maximum(node.column_indices)
# end

# importing issorted to overload it. Okay, since only involving custom types
import Base.issorted
function issorted(node::IndexSuperNode)
    return issorted(node.row_indices) && issorted(node.column_indices)
end

function issorted(assignment::IndirectSupernodalAssignment) 
    # Checking whether each supernode is sorted, and whether the supernodes are sorted according to their first index.
    all(issorted.(assignment.supernodes)) && issorted(first.(getfield.(assignment.supernodes, :column_indices)))
end