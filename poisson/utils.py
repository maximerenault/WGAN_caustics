#MIT License
#
#Copyright (c) 2022 Maxime Renault
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

import numpy as np
import dolfin as dlf
import scipy
import shapely.geometry as geom
import PIL



def get_area_matrix(grid, tri: bool):
    # Gives out the matrix of the areas of 
    # the triangles composing the mesh
    #
    # grid  :   positions of grid points in order
    # tri   :   1 to compute oriented triangles areas
    #           0 to compute oriented quads areas

    h_1, w_1 = grid.shape[:-1]
    x1 = grid[:-1,:-1,:1]
    x2 = grid[:-1,1:,:1]
    x3 = grid[1:,1:,:1]
    x4 = grid[1:,:-1,:1]
    y1 = grid[:-1,:-1,1:]
    y2 = grid[:-1,1:,1:]
    y3 = grid[1:,1:,1:]
    y4 = grid[1:,:-1,1:]
    if tri :
        areas_tris1 = (-(x1*y2+x2*y4+x4*y1-y1*x2-y2*x4-y4*x1)/2.).reshape(h_1-1,w_1-1)
        areas_tris2 = (-(x2*y3+x3*y4+x4*y2-y2*x3-y3*x4-y4*x2)/2.).reshape(h_1-1,w_1-1)
        out = np.concatenate((areas_tris1,areas_tris2))
    else :
        out = (-(x1*y2+x2*y3+x3*y4+x4*y1-y1*x2-y2*x3-y3*x4-y4*x1)/2.).reshape(h_1-1,w_1-1)
    return out



def gen_mesh(grid, cells):
    # Generates a Fenics mesh from
    # points and cells
    #
    # grid  :   positions of grid points in order
    # cells :   cells of the mesh defined by indices of points

    n_nodes = len(grid)
    n_cells = len(cells)
    mesh = dlf.Mesh()
    editor = dlf.MeshEditor()
    editor.open(mesh, 'triangle', 2, 2)
    editor.init_vertices(n_nodes)
    editor.init_cells(n_cells)

    for i in range(n_nodes) :
        editor.add_vertex(i, grid[i])
    for i in range(n_cells) :
        editor.add_cell(i, cells[i])
    editor.close()

    return mesh



def voronoi(h : int, w : int, points):
    # Returns a Voronoi diagram constrained
    # on the borders with symetry

    # Mirror points
    points_center = np.copy(points).reshape((h+1)*(w+1),2)
    # sides
    points_left = np.copy(points[:,1,:]).reshape(h+1,2)
    points_left[:, 1] = points[0,0,1] - (points_left[:, 1]-points[0,0,1])
    points_right = np.copy(points[:,-2,:]).reshape(h+1,2)
    points_right[:, 1] = points[0,-1,1] - (points_right[:, 1]-points[0,-1,1])
    points_down = np.copy(points[-2,:,:]).reshape(w+1,2)
    points_down[:, 0] = points[-1,0,0] - (points_down[:, 0]-points[-1,0,0])
    points_up = np.copy(points[1,:,:]).reshape(w+1,2)
    points_up[:, 0] = points[0,0,0] - (points_up[:, 0]-points[0,0,0])
    # corners
    points_corners = np.array([points[1,1,:], points[1,-2,:], points[-2,-2,:], points[-2,1,:]])
    points_corners[0] = points[0,0,:] - (points_corners[0]-points[0,0,:])
    points_corners[1] = points[0,-1,:] - (points_corners[1]-points[0,-1,:])
    points_corners[2] = points[-1,-1,:] - (points_corners[2]-points[-1,-1,:])
    points_corners[3] = points[-1,0,:] - (points_corners[3]-points[-1,0,:])
    # Concatenate all points
    points = np.concatenate((points_center,points_left,points_right,points_down,points_up,points_corners),axis=0)
    # Compute Voronoi
    vor = scipy.spatial.Voronoi(points)
    ## Filter regions
    idx_list = np.array(vor.point_region[:(h+1)*(w+1)], dtype=int)
    regions = np.array(vor.regions, dtype=object)
    vor.filtered_regions = regions[idx_list]
    vor.filtered_points = points_center

    return vor
    


def influence_area(h : int, w : int, grid):
    # Returns the value of the influence
    # area of each point on the grid

    vor = voronoi(h, w, grid)
    polys = [geom.Polygon(vor.vertices[region]) for region in vor.filtered_regions]
    areas = np.array([poly.area for poly in polys]).reshape(h+1,w+1)
    # sides and corners
    areas[:,0] *= 0.5
    areas[:,-1] *= 0.5
    areas[0,:] *= 0.5
    areas[-1,:] *= 0.5

    return areas



def integrate_img(h : int, w : int, image_grey : PIL.Image.Image) :
    # Returns an approximation of the integral
    # of the image for each point on the grid

    img_resized = image_grey.resize((w*2,h*2))
    img_resized = np.array(img_resized,dtype=float)
    integ_im = np.zeros((h+1,w+1))
    #corners
    integ_im[0,0] = img_resized[0,0]
    integ_im[0,-1] = img_resized[0,-1]
    integ_im[-1,-1] = img_resized[-1,-1]
    integ_im[-1,0] = img_resized[-1,0]
    #sides
    integ_im[1:-1,0] = img_resized[1:-1:2,0]+img_resized[2:-1:2,0]
    integ_im[1:-1,-1] = img_resized[1:-1:2,-1]+img_resized[2:-1:2,-1]
    integ_im[-1,1:-1] = img_resized[-1,1:-1:2]+img_resized[-1,2:-1:2]
    integ_im[0,1:-1] = img_resized[0,1:-1:2]+img_resized[0,2:-1:2]
    #inside
    integ_im[1:-1,1:-1] = img_resized[1:-1:2,1:-1:2]+img_resized[2:-1:2,1:-1:2]+img_resized[1:-1:2,2:-1:2]+img_resized[2:-1:2,2:-1:2]
    
    return integ_im



def find_dt(grid, grad):
    # Returns the highest dt possible without
    # inverting any element (dichotomy)

    h_1, w_1 = grid.shape[:-1]
    dt1 = 1000.
    dt0 = 0.
    eps = dt1-dt0
    while eps>0.0001 :
        a_min1 = np.min(get_area_matrix(grid+dt1*grad, True))
        a_min0 = np.min(get_area_matrix(grid+dt0*grad, True))
        if abs(a_min1) > abs(a_min0) :
            dt1 = (dt1+dt0)/2.
        else :
            dt0 = (dt1+dt0)/2.
        eps = dt1-dt0
    return dt1



def solve_poisson(h : int, w : int, lens_mesh, loss, voron, source_to_div = 0, out_grad = True):
    # Solves Poisson equation 

    # Define function space
    P1      = dlf.FiniteElement("Lagrange", lens_mesh.ufl_cell(), 1)
    V_u     = dlf.FunctionSpace(lens_mesh, P1)
    R       = dlf.FiniteElement("Real", lens_mesh.ufl_cell(), 0)
    V_uc    = dlf.FunctionSpace(lens_mesh, P1 * R)

    W = dlf.VectorFunctionSpace(lens_mesh, "CG", 1)

    vxmap = dlf.vertex_to_dof_map(V_u)
    wxmap = dlf.vertex_to_dof_map(W)

    # Define variational variables
    (u,c) = dlf.TrialFunction(V_uc)
    (v,d) = dlf.TestFunction(V_uc)

    if source_to_div == 0 :
        f = dlf.Function(V_u)
        f = dlf.fem.interpolation.interpolate(f, V_u)

        # Define source term at its nodes
        ls = np.array([int(i / (w+1)) for i in range(V_u.dim())])
        ks = np.array([i % (w+1) for i in range(V_u.dim())])
        if voron :
            f.vector()[vxmap[:]] = loss[ls,ks]
        else :
            f.vector()[vxmap[:]] = loss(ls,ks,grid=False)[:]
    else :
        source_proj = dlf.Function(W)
        source_proj.vector()[wxmap] = source_to_div.reshape((h+1)*(w+1)*2)[:]
        div_source = dlf.project(dlf.div(source_proj),V_u)
        f = div_source

    # Variational formulation
    a = (dlf.inner(dlf.grad(u), dlf.grad(v)) + c*v + u*d)*dlf.dx
    L = f*v*dlf.dx

    # Solution Function
    u = dlf.Function(V_uc)

    # Compute solution
    dlf.solve(a == L, u)
    (u, c) = u.split()

    outs = [u, vxmap]

    if out_grad :
        # Gradient
        gradu = dlf.project(dlf.grad(u), W)
        lens_grad = np.array(gradu.vector().get_local()[wxmap]).reshape(h+1,w+1,2)
        outs.append(lens_grad)

    return outs