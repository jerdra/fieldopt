#!/usr/bin/env python
## Minimal test suite for validating tetrahedral projection algorithm
##
## Tests implemented:
##
##     1. Fully embedded tetrahedrons
##     2. Bordering sample rejection and resampling
##     3. Direction-specific dual membership
##     4. Single voxel exclusion test (2x2 voxels, planar, 1 voxel excluded)
##     5. Comparison to analytical solution for simple 2 voxel test
##


import os
import numpy as np
from fieldopt import tetrapro

import mock

def gen_fully_embedded_tet(n,shape):
    '''
    Generate <n> Class 1 (fully embedded) tetrahedrons within a data grid of shape <shape>
    Returns a list of coordinates for each node as well as vertex ID for each tetrahedron

    Implementational Details:
    For each tetrahedron, select a voxel i,j,k in linear ordering
    Generate a (4x3) matrix of uniformly distributed variables, where n=4 describes the number of vertices
        and m=3 describes the i,j,k coordinates respectively.

    Adjust columns of the random matrix according to voxel identity to ensure complete embedding of
    tetrahedral vertices in voxel i,j,k
    '''

    node_ids = np.ones((n,4), dtype=np.int).cumsum().reshape((n,4)) - 1
    coord_array = np.zeros((n*4,3))
    for i in np.arange(0,n):

        #Select voxel in shape
        step_z = i // (shape[0] * shape[1])
        step_y = i // (shape[0]) - shape[2]*step_z
        step_x = i - shape[1]*step_y - shape[1]*shape[2]*step_z
        selected_vox=(step_x,step_y,step_z)

        #Generate 4 sets of random integers within the boundaries defined by voxel
        rand_coords = np.random.random(size=(4,3))

        #Modify each value generated by boundariers defined by voxel
        rand_coords[:,0] += step_x
        rand_coords[:,1] += step_y
        rand_coords[:,2] += step_z

        coord_array[4*i,:] = rand_coords[0,:]
        coord_array[4*i+1,:] = rand_coords[1,:]
        coord_array[4*i+2,:] = rand_coords[2,:]
        coord_array[4*i+3,:] = rand_coords[3,:]

    return node_ids, coord_array.flatten()

def test_fully_embedded_tets_for_projection():
    '''
    Fully embedded tetrahedron test, full algorithm testing
    '''

    N = 10
    t = N**3
    affine = np.eye(4)
    data_grid = np.ones((N**3), dtype=np.int64).cumsum()
    data_grid = data_grid.reshape((N,N,N))
    data_grid = np.swapaxes(data_grid, 0, 2)

    node_list, coord_array = gen_fully_embedded_tet(t, data_grid.shape)
    est = tetrapro.tetrahedral_projection(node_list, coord_array, data_grid, affine, n_iter=1000)
    total_embedding_score = np.max(est, axis=0).sum()
    assert int(total_embedding_score) == t


def test_directional_membership_is_correct():
    '''
    Generate a 3x3x3 cube data grid then test memberships in the following order
    [-x, +x, -y, +y, -z, +z]

    Orientation Info:
    Since voxel array coordinates don't natively match with the coordinate space used internally
    we've implicitly transformed the data when assigning memberships in datagrid. 

    Notice that in c_{direction} (x:LR, y:UD, z:IO}) 
    Then notice that in data_grid X is interpreted to mean LR instead of the usual UD in array format
    '''

    #Node IDs are the same across coordinate sets
    node_ids = np.array([
        [0,1,2,3],
        [0,1,2,4]
        ])

    #Make planar fixed coordinates for each pair
    c_updown = np.array([
        [1, 1.5, 2],
        [2, 1.5, 2],
        [1.5, 1.5, 1],
        [1.5, 0, 1.5], #top
        [1.5, 2.8, 1.5], #bottom
    ]).flatten()

    c_leftright = np.array([

        [1.5, 1, 2],
        [1.5, 2, 2],
        [1.5, 1.5, 1],
        [0, 1.5, 1.5], #left
        [2.8, 1.5, 1.5] #right
    ]).flatten()

    c_inout = np.array([

        [1, 2, 1.5],
        [2, 2, 1.5],
        [1.5, 1, 1.5],
        [1.5, 1.5, 0], #front
        [1.5, 1.5, 2.8] #back
    ]).flatten()

    #Make data grid and write values in testing region
    data_grid = np.zeros( (3,3,3), dtype=np.int64)

    #Central point
    data_grid[1,1,1] = 0

    #Test points
    data_grid[1,0,1] = 1 #top
    data_grid[1,2,1] = 2 #bottom
    data_grid[0,1,1] = 3 #left
    data_grid[2,1,1] = 4 #right
    data_grid[1,1,0] = 5 #front
    data_grid[1,1,2] = 6 #back

    #Basic affine
    affine = np.eye(4)

    #Run projection algorithm
    ud_est = tetrapro.tetrahedral_projection(node_ids, c_updown, data_grid, affine, n_iter=1000)
    lr_est = tetrapro.tetrahedral_projection(node_ids, c_leftright, data_grid, affine, n_iter=1000)
    io_est = tetrapro.tetrahedral_projection(node_ids, c_inout, data_grid, affine, n_iter=1000)

    #Test each
    assert ud_est[0,1] != 0
    assert ud_est[1,2] != 0
    assert lr_est[0,3] != 0
    assert lr_est[1,4] != 0
    assert io_est[0,5] != 0
    assert io_est[1,6] != 0

def test_single_voxel_exclusion():
    '''
    Given a 2X2X1 slab of voxels, the single voxel exclusion test tests the
    assertion that a tetrahedron with nodes rooted at two corner points and
    opposing edge mid-points will be embedded in 3/4 voxels. The
    voxel containing 0 nodes should have no tetrahedral volume but will
    always be included in the bounding box enclosure.

    Orientation Details:
    In voxel/array space X refers to rows, Y refers to columns and Z is depthwise
    In coordinate space the roles of X and Y are switched. X is the LR axis, Y is the SI axis

    The affine transform solves this discrepancy between coordinate space and array/voxel space
    '''

    #Make slab
    data_grid = np.ones( (2,2,1), dtype=np.int)

    #Set exclusion voxel (top right)
    data_grid[0,1,0] = 2

    #Single tetrahedron
    tet_nodes = np.array([
        [0,1,2,3]
        ], dtype=np.int)

    #Set node coordinates
    tet_coords = np.array([
        [1.999,1.999,0.999],    #2 nodes at each bottom-right corner
        [1.999,1.999,0],
        [0, 1.999, 0.5],        #1 node on bottom-left edge midpoint
        [0, 0, 0.5]             #1 node on top-left edge midpoint
    ], dtype=np.float64).flatten()

    #Affine transform from array --> coordinate ordering
    affine = np.array([
        [0,1,0,0],
        [1,0,0,0],
        [0,0,1,0],
        [0,0,0,1]
        ], dtype=np.float64)

    #Run projection
    est = tetrapro.tetrahedral_projection(tet_nodes, tet_coords, data_grid, affine, n_iter=1000)
    assert est[0,2] == 0

def test_monte_carlo_approximates_analytical_solution_simple_case():
    '''
    Given a 2X1X1 column of voxels, the monte carlo approximation test
    tests the assertion that using a tetrahedral random sampling method
    approximates the true analytical volume of the embedded tetrahedron to
    some epsilon as the number of monte carlo samples approaches very large
    N
    '''

    #TODO - Derive the analytical solution to this problem

    pass
