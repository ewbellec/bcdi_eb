import numpy as np
import pylab as plt

import gif

import sys
sys.path.append('/data/id01/inhouse/bellec/software/sharedipynb/gitlab/bcdi_eb/')
from Plot_utilities import *
from Global_utilities import *

######################################################################################################################################
##################################             module phase strain gif                ################################################
###################################################################################################################################### 


@gif.frame
def Gif_one_slice_plot(module,phase,strain, axis, indice,
                       voxel_sizes=None,
                       vmin_phase=None, vmax_phase=None,
                       vmax_strain=None,
                       fw=4):
    fig,ax = plt.subplots(1,3, figsize=(fw*3,fw))
    imgs = []

    shape = module.shape

    vmin_mod = np.nanmin(module)
    vmax_mod = np.nanmax(module)

    if vmin_phase is None:
        vmin_phase = np.nanmin(phase)
    if vmax_phase is None:
        vmax_phase = np.nanmax(phase)
        
    if vmax_strain is None:
        vmax_strain = np.nanmax(np.abs(strain))
    vmin_strain = -vmax_strain
    
    if voxel_sizes is not None:
        voxel_sizes = .1*np.array(voxel_sizes) # Put the voxel_sizes in nanometers
        extent = [[0, module.shape[2]*voxel_sizes[2], 0, module.shape[1]*voxel_sizes[1]], 
                   [0, module.shape[2]*voxel_sizes[2], 0, module.shape[0]*voxel_sizes[0]],
                   [0, module.shape[1]*voxel_sizes[1], 0, module.shape[0]*voxel_sizes[0]]]
    else:
        extent=[None,None,None]


    imgs.append(ax[0].matshow(module.take(indices=indice, axis=axis), cmap='gray_r', vmin=vmin_mod, vmax=vmax_mod,
                              extent=extent[axis]))
    imgs.append(ax[1].matshow(phase.take(indices=indice, axis=axis), cmap='hsv', vmin=vmin_phase, vmax=vmax_phase,
                              extent=extent[axis]))
    imgs.append(ax[2].matshow(strain.take(indices=indice, axis=axis), cmap='coolwarm', vmin=vmin_strain, vmax=vmax_strain,
                              extent=extent[axis]))

    add_colorbar_subplot(fig,ax,imgs)
    
    if voxel_sizes is not None:
        for axe in ax:
            axe.xaxis.set_ticks_position('bottom')
            axe.set_xlabel('nm',fontsize=15)

    fig.tight_layout()
    return

def make_gif_module_phase_strain(module,phase,strain, axis,
            voxel_sizes=None,
            vmin_phase=None,vmax_phase=None,
            vmax_strain=None,
            fw=4, name='gif_default_name', duration=10):
    
    frames = [Gif_one_slice_plot(module,phase,strain,axis, indice,voxel_sizes=voxel_sizes, fw=fw,
                                 vmin_phase=vmin_phase, vmax_phase=vmax_phase, vmax_strain=vmax_strain) for indice in range(module.shape[axis])]
    gif.save(frames, name+'.gif', duration=duration, unit='s', between='startend')
    print('gif saved in : ',name+'.gif')
    return



######################################################################################################################################
##################################             Rotating isosurface gif                ################################################
######################################################################################################################################

from skimage import measure

@gif.frame
def gif_one_rotation(verts , faces,  angle):
    
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2],
                    cmap='gray', lw=1)
    # Hide grid lines
    ax.grid(False)

    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.view_init(20, angle)
    
    fig.tight_layout()
    return

def make_gif_pixels_3d(module_ortho,
                       iso_val=0.2,
                       name='gif_default_name', duration=10, nb_angles=30):
    
    verts , faces, normals, values = measure.marching_cubes(module_ortho, iso_val, spacing=(0.1, 0.1, 0.1))
    angle_list = np.linspace(0,360,nb_angles) 
    frames = [gif_one_rotation(verts , faces,  angle) for angle in angle_list]
    gif.save(frames, name+'.gif', duration=duration, unit='s', between='startend')
    print('gif saved in : ',name+'.gif')
    
    return


######################################################################################################################################
#################################             Gif from a list of images                ###############################################
######################################################################################################################################

from PIL import Image

@gif.frame
def Gif_one_image(file,
                  fw=6):
    
    img = Image.open(file)
    ratio = img.size[0]/img.size[1]
    fig,ax = plt.subplots(1,figsize=(fw*ratio,fw))
    ax.imshow(img)
    return

def make_gif_list_images(files, name='gif_default_name', duration=10,
                         fw=6):
    
    frames = [Gif_one_image(file,fw=fw) for file in files]
    gif.save(frames, name+'.gif', duration=duration, unit='s', between='startend')
    print('gif saved in : ',name+'.gif')
    return