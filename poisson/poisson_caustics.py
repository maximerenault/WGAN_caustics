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
import PIL
import trimesh
import dolfin as dlf
import matplotlib.pyplot as plt
import scipy
import utils

class CausticsDesign:
    lens_grid = np.array([])
    lens_cells = np.array([])
    lens_mesh = 0
    loss = np.array([])
    voron = False
    w = 0
    h = 0
    elem_size = 0.
    it_nb = 10
    screen_dist = 2.

    def __init__(self, file_name : str, image_grey : PIL.Image.Image, size : float) -> None:
        #   file_name   : the name of this lens for export
        #   image_grey  : your image you want to create a lens for
        #   size        : width of the physical lens in meters

        self.file_name = file_name
        self.img_grey = image_grey
        self.size = size
        pass

    def generate_grid_mesh(self, grid_res : int) :
        # Generates a grid in the form of a numpy array,
        # as well as the corresponding dolfin mesh
        #
        #   grid_res     : resolution of the grid in width

        self.w, self.h = self.img_grey.size
        self.h = int((self.h/self.w)*grid_res)
        self.w = grid_res
        self.elem_size = self.size/self.w

        self.lens_grid = np.array(np.meshgrid(np.linspace(0, self.elem_size*self.h, self.h+1, dtype = np.float64),\
                                              np.linspace(0, self.elem_size*self.w, self.w+1, dtype = np.float64))).T.reshape(self.h+1,self.w+1,2)

        self.lens_cells = np.array([[[[i, i+self.w+2, i+self.w+1],[i, i+1, i+self.w+2]] for i in range((self.w+1)*j,self.w+(self.w+1)*j)] for j in range(0,self.h)]).reshape(self.h*self.w*2,3)

        self.lens_mesh = utils.gen_mesh(self.lens_grid.reshape((self.h+1)*(self.w+1),2), self.lens_cells)

        return

    def compute_loss(self, voron : bool) :
        # Returns the loss between influence areas
        # and light values corresponding to these points

        self.voron = voron

        if self.voron :
            source  = utils.integrate_img(self.h, self.w, self.img_grey)
            areas   = utils.influence_area(self.h, self.w, self.lens_grid)
        else :
            source  = self.img_grey
            areas   = utils.get_area_matrix(self.lens_grid, False)
        
        allbright = np.sum(source)
        loss = areas/(self.elem_size**2)-source*(self.h*self.w)/allbright
        self.loss = (loss-np.mean(loss))/self.w
        return

    def show_image(self) :
        npimg = np.array(self.img_grey,dtype=float)
        plt.imshow(npimg, cmap='gray')
        return

    def show_loss(self) :
        source = plt.imshow(self.loss, cmap='gray')
        plt.colorbar(source)
        return

    def compute_transport_map(self, it_nb : int) :
        # Solves Poisson's equation to map
        # the light source to the light goal

        self.it_nb = it_nb
        max_l = np.max(self.loss)
        min_l = np.min(self.loss)
        tot_l = np.sum(np.abs(self.loss))
        compt = 0

        if self.voron :
            use_loss = self.loss
        else :
            use_loss = scipy.interpolate.RectBivariateSpline(np.linspace(0.5, self.h-0.5, self.h, dtype = np.float64),\
                                                             np.linspace(0.5, self.w-0.5, self.w, dtype = np.float64), self.loss)

        while compt < self.it_nb :
            print("Max error : %0.4f Min error : %0.4f Total : %0.4f" %(max_l, min_l, tot_l))
            u, vxmap, lens_grad = utils.solve_poisson(self.h, self.w, self.lens_mesh, use_loss, self.voron)
            lens_grad[:,0,1] = 0
            lens_grad[:,-1,1] = 0
            lens_grad[0,:,0] = 0
            lens_grad[-1,:,0] = 0

            dt = utils.find_dt(self.lens_grid, lens_grad)*0.2
            self.lens_grid += dt*lens_grad

            self.lens_mesh = utils.gen_mesh(self.lens_grid.reshape((self.h+1)*(self.w+1),2), self.lens_cells)
            self.loss = self.compute_loss(self.voron)
            max_l = np.max(self.loss)
            min_l = np.min(self.loss)
            tot_l = np.sum(np.abs(self.loss))

            if not self.voron :
                use_loss = scipy.interpolate.RectBivariateSpline(np.linspace(0.5, self.h-0.5, self.h, dtype = np.float64),\
                                                                 np.linspace(0.5, self.w-0.5, self.w, dtype = np.float64), self.loss)
            compt += 1

        return
    
    def compute_height(self, rho = 1.49, screen_dist = 2.) :
        self.screen_dist = screen_dist
        h_xy = np.zeros((self.h+1, self.w+1))
        screen_coord = np.array(np.meshgrid(np.linspace(0, self.elem_size*self.h, self.h+1,dtype = np.float64),\
                                            np.linspace(0, self.elem_size*self.w, self.w+1,dtype = np.float64))).T.reshape(self.h+1, self.w+1,2)
        dist_grid = screen_coord-self.lens_grid

        for i in range(20) :
            # Calculate normals
            k_xy = rho*np.sqrt(np.square(np.linalg.norm(dist_grid, axis=2))+np.square(self.screen_dist-h_xy))-(self.screen_dist-h_xy)
            N_xy = dist_grid/(k_xy.reshape(self.h+1,self.w+1,1))

            u, vxmap = utils.solve_poisson(self.h, self.w, self.lens_mesh, self.loss, self.voron, source_to_div = N_xy, out_grad = False)

            if np.max(h_xy-u.vector().get_local()[vxmap].reshape(self.h+1,self.w+1))>10e-6 :
                h_xy = u.vector().get_local()[vxmap].reshape(self.h+1,self.w+1)
            else : break
        
        self.height = h_xy
        return
    
    def save_mesh(self):
        ## Saving mesh to stl

        thickness = self.size/10.     # Distance between back of the object and plane z = 0

        elem_size = self.elem_size
        h = self.h
        w = self.w

        base_grid = np.array(np.meshgrid(np.linspace(0,elem_size*h,h+1,dtype = np.float64),np.linspace(0,elem_size*w,w+1,dtype = np.float64))).T.reshape(h+1,w+1,2)
        base_z = np.ones((h+1,w+1))*thickness
        base_vertices = np.concatenate((base_grid,base_z.reshape(h+1,w+1,1)),axis=2).reshape((h+1)*(w+1),3)

        # Defining cells
        base_cells = np.array([[[[i, i+w+1, i+w+2],[i, i+w+2, i+1]] for i in range((w+1)*j,w+(w+1)*j)] for j in range(h)]).reshape(h*w*2,3) + (h+1)*(w+1)
        # Top, bottom, left, right
        border_cells0 = np.array([[[i, i+(w+1)*(h+1), i+1+(w+1)*(h+1)],[i, i+1+(w+1)*(h+1), i+1]] for i in range(w)]).reshape(w*2,3)                    
        border_cells1 = np.array([[[i, i+1+(w+1)*(h+1), i+(w+1)*(h+1)],[i, i+1, i+1+(w+1)*(h+1)]] for i in range(w)]).reshape(w*2,3) + h*(w+1)          
        border_cells2 = np.array([[[i*(w+1), (w+1)*(h+i+2), (w+1)*(h+i+1)],[i*(w+1), (i+1)*(w+1), (w+1)*(h+i+2)]] for i in range(h)]).reshape(h*2,3)    
        border_cells3 = np.array([[[i*(w+1), (w+1)*(h+i+1), (w+1)*(h+i+2)],[i*(w+1), (w+1)*(h+i+2), (i+1)*(w+1)]] for i in range(h)]).reshape(h*2,3) + w

        lens_vertices = np.concatenate((self.lens_grid, self.height.reshape(h+1,w+1,1)),axis=2).reshape((h+1)*(w+1),3)

        # Concatenate everything to build one mesh
        vertices = np.concatenate((lens_vertices, base_vertices))
        cells = np.concatenate((self.lens_cells, base_cells, border_cells0, border_cells1, border_cells2, border_cells3))

        # Rotate and move to use in Blender
        vertices[:, [2, 0]] = vertices[:, [0, 2]]
        vertices[:,2] *= -1
        vertices[:, [1, 0]] = vertices[:, [0, 1]]
        vertices[:,1] *= -1
        z_min = np.min(vertices[:,2])
        vertices += np.array([[-0.5,0,-z_min]])

        final_mesh = trimesh.Trimesh(vertices, cells)
        final_mesh.export(self.file_name+"_"+str(w)+"_mesh.stl")
        return