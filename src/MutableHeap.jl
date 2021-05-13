import LinearAlgebra.sortperm

#############################################################################
#Implementation of a mutable maximal Heap
#############################################################################
#Struct representing a node of the heap
abstract type AbstractNode{Tval, Tid} end

struct Node{Tval, Tid} <: AbstractNode{Tval, Tid}
  val::Tval
  id::Tid
end

struct RankedNode{Tval, Tid, Trank} <: AbstractNode{Tval, Tid}
  val::Tval
  id::Tid
  #larger ranks get picked last
  rank::Trank
end

function vtype(node::AbstractNode)
  return typeof(node.val)
end

function vtype(type::Type{<:AbstractNode{Tval,Tid}}) where {Tval,Tid}
  return Tval
end


function getval(node::AbstractNode)
  return node.val
end

# creates a new node with a different value
function setval(node::Node, val)
  return typeof(node)(val, getid(node))
end

function idtype(node::AbstractNode)
  return typeof(node.id)
end

function idtype(type::Type{<:AbstractNode{Tval,Tid}}) where {Tval,Tid}
  return Tid
end

function getid(node::AbstractNode)
  return node.id
end

function ranktype(node::RankedNode)
  return typeof(node.rank)
end

function getrank(node::RankedNode)
  return node.rank
end

#Mutable Heap (maximal value first)
struct MutableHeap{Tn<:AbstractNode,Tid}
  # Vector containing the nodes of the heap
  nodes::Vector{Tn}
  # Vector providing a lookup for the nodes
  lookup::Vector{Tid}
  # We make sure none of the input nodes has the typmin of Tval, since this this value is reserved to implement popping from the heap
  function MutableHeap{Tn, Tid}(nodes::Vector{Tn}, lookup::Vector{Tid}) where {Tn<:AbstractNode,Tid} 
    !all(getval.(nodes) .> typemin(vtype(Tn))) ? error("typemin of value type reserved") : new{Tn,Tid}(nodes, lookup) 
  end
end

function MutableHeap(values::AbstractVector)
  Tval = eltype(values)
  Tid = Int
  nodes = Vector{Node{Tval,Tid}}(undef, length(values))
  lookup = Vector{Int}(undef, length(values))
  for (id, val) in enumerate(values)
    nodes[id] = Node{Tval,Tid}(val, id)
    lookup[id] = id
  end
  perm = sortperm(nodes,rev=true)
  nodes .= nodes[perm]
  lookup .= lookup[perm]
  return MutableHeap{eltype(nodes), eltype(lookup)}(nodes, lookup)
end

function nodetype(h::MutableHeap{TNode}) where TNode
  return TNode
end

#A function to swap two heapNodes in 
function _swap!(h,
                a,
                b)
  # Assigning the new values to the lookup table 
  h.lookup[ h.nodes[a].id ] = b
  h.lookup[ h.nodes[b].id ] = a
  tempNode = h.nodes[a]
  h.nodes[a] = h.nodes[b]
  h.nodes[b] = tempNode
end

#Node comparisons
import Base.isless
function isless( a::RankedNode, b::RankedNode)
  isless((-a.rank, a.val), (-b.rank, b.val))
end

function isless( a::Node, b::Node)
  isless(a.val, b.val)
end


import Base.>=
function >=(a::Node, b::Node)
  a.val >= b.val
end

function >=(a::RankedNode, b::RankedNode)
  (-a.rank, a.val) >= (-b.rank, b.val)
end

# presently not used
# function >=(a::Node, b) 
#   a.val >= b
# end

import Base.>
function >(a::RankedNode, b::RankedNode) 
  (-a.rank, a.val) > (-b.rank, b.val)
end

function >(a::AbstractNode, b::AbstractNode) 
  a.val > b.val
end

# function >( a::Node{Tv,Ti}, b::Tv ) where {Tv,Ti}
#   a.val > b
# end

#Function that looks at element h.nodes[hInd] and moves it down the tree 
#if it is sufficiently small. Returns the new index if a move took place, 
#and lastindex(h.nodes), else
function _moveDown!(h::MutableHeap, hInd)
  pivot = h.nodes[hInd]
  #If both children exist:
  if 2 * hInd + 1 <= lastindex( h.nodes )
    #If the left child is larger:
    if h.nodes[2 * hInd] >= h.nodes[ 2 * hInd + 1]
      #Check if the child is larger than the parent:
      if h.nodes[2 * hInd] >= pivot
        _swap!( h, hInd, 2 * hInd )
        return 2 * hInd
      else
        #No swap occuring:
        return lastindex( h.nodes )
      end
    #If the left child is larger:
    else
      #Check if the Child is larger than the paren:
      if h.nodes[2 * hInd + 1] >= pivot
        _swap!( h, hInd, 2 * hInd + 1 )
        return  2 * hInd + 1
      else
        #No swap occuring:
        return lastindex( h.nodes )
      end
    end
    #If only one child exists:
  elseif 2 * hInd <= lastindex( h.nodes )
    if h.nodes[2 * hInd] > pivot
      _swap!( h, hInd, 2 * hInd )
      return 2 * hInd 
    end
  end
  #No swap occuring:
  return lastindex( h.nodes )
end

#Get the leading node
function top_node(h::MutableHeap)
  return first(h.nodes)
end

#Gets the leading node and moves it to the back
function top_node!(h::MutableHeap{Node{Tv,Ti},Ti}) where {Tv, Ti}
  out = first(h.nodes)
  # move the node to the very bottom of the list
  update!(h, getid(out), typemin(Tv))
  return out 
end

#Updates (decreases) an element of the heap and restores the heap property
function update!(h::MutableHeap{Node{Tv,Ti},Ti}, id::Ti, val::Tv) where {Tv,Ti}
  tempInd::Ti = h.lookup[id]
  if h.nodes[tempInd].val > val
    h.nodes[tempInd] = setval(h.nodes[tempInd], val)
    while ( tempInd < lastindex( h.nodes ) )
      tempInd = _moveDown!(h, tempInd)
    end
    return val
  else
    return h.nodes[id].val
  end
end