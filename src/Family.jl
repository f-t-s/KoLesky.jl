using StaticArrays

struct Member{Tval, Tid}
  val::Tval
  id::Tid
end

import Base.isless
function isless(a::Member{Tval, Tid}, b::Member{Tval, Tid}) where{Tval, Tid}
  return isless((a.val, a.id), (b.val, b.id))
end 

struct Family{Tval, Tid}
  n_columns::MVector{1, Tid}
  n_nz ::MVector{1, Tid}
  n_buffer::MVector{1, Tid}

  # This array gives for contains the ordering. The i-th parent in the 
  #daycare has id P[i]
  P::Vector{Tid}

  # This array gives for contains the inverse ordering.
  revP::Vector{Tid}

  # The array that contains the first "child" for every parent
  colptr::Vector{Tid}

  # The array that contains the global id-s of the children 
  rowval::Vector{Member{Tval,Tid}}

  # parents[id] contains a member struct specifying the distance of id to its parent and the position in the ordering of the parent
  parents::Vector{Member{Tval,Tid}}
end

function Family{Tval,Tid}(dofs::AbstractArray{<:AbstractDOF}) where {Tval,Tid}
  N = length(dofs)
  return  Family{Float64, Int64}(MVector(zero(Tid)), MVector(zero(Tid)), MVector(N), zeros(Tid, N), zeros(Tid, N), zeros(Tid, N + one(Tid)), Vector{Member{Tval, Tid}}(undef, N), Vector{Member{Tval, Tid}}(undef, N)) 
end

function Family(dofs::AbstractArray{Point{d,Tval}}) where {d, Tval}
  return  Family{Tval, Int}(dofs)
end

#Function that begins a new parent aka column in daycare
function new_column!(f::Family, Id)
  f.n_columns[1] += 1 
  f.P[f.n_columns[1]] = Id
  f.revP[Id] = f.n_columns[1]
  f.colptr[f.n_columns[1]] = f.n_nz[1] + 1
  f.colptr[f.n_columns[1] + 1] = f.n_nz[1] + 1
end

function new_child!(f::Family, new_child) 
  # If the buffer is flowing over, increase it
  if f.n_nz[1] >= f.n_buffer[1]
    if f.n_nz[1] <= 1e6
      f.n_buffer[1] = 2 * f.n_buffer[1]
    else 
      f.n_buffer[1] = f.n_buffer[1] + 1e6
    end
    resize!( f.rowval, f.n_buffer[1] )
  end
  f.n_nz[1] += 1
  f.colptr[f.n_columns[1] + 1] += 1
  f.rowval[f.n_nz[1]] = new_child 
end

function new_children!(f::Family, new_children)
  # If the buffer is flowing over, increase it
  while f.n_nz[1] + size(new_children,1) >= f.n_buffer[1] - 1
    if f.n_nz[1] <= 1e6
      f.n_buffer[1] = 2 * f.n_buffer[1]
    else 
      f.n_buffer[1] = f.n_buffer[1] + 1e6
    end
    resize!(f.rowval, f.n_buffer[1])
  end

  f.n_nz[1] += size(new_children,1)
  f.colptr[f.n_columns[1] + 1] += size(new_children,1)
  f.rowval[f.n_nz[1] - size(new_children,1) + 1 : f.n_nz[1]] .= new_children
end

function column_iterator(f, column_index)
  return f.colptr[column_index] : (f.colptr[column_index + 1] - 1)
end
