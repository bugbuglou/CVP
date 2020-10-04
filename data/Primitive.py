# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function
import sys
sys.path.append('C:/Users/john/mydir/CVP')
sys.path.append('C:/Users/john/mydir/CVP/utils')
import glob
import os
import json
import numpy as np
# import debug_init_paths
import torch
import torchvision.transforms as T
from PIL import Image
import h5py
import pandas as pd
import io
from torch.utils.data import Dataset, DataLoader
from utils.data_utils import imagenet_preprocess, Resize

def tell(list, array):
    id = True
    for k in list:
        if np.all(k==array):
            id = False
            break
    return id


def raw_to_agent_particles(particles, Vmat):
    '''
    Map raw particle coordinates to agent-relative coordinates

    #### INPUTS ####
    raw_particles: [B,T,N,3] tensor of N particle (x,y,z) world coordinates
    Vmat: [4,4] matrix that transforms raw to agent-relative homogeneous coordinates

    #### RETURNS ####
    agent_relative_particles: [B,T,N,3] agent-relative coordinates of particles
    '''
    shape = list(np.shape(particles))
    if len(shape) != 4:
        particles = np.reshape(particles, (1,1,-1,3))
    shape = list(np.shape(particles))
    Vmat = np.zeros([shape[0],shape[1],4,4], np.float32) + np.reshape(Vmat, [1,1,4,4])
    raw_particles = np.concatenate([particles, np.ones(shape[0:3]+[1], dtype=np.float32)],
                              axis=3) # [B,T,N,4]
    raw_particles = np.transpose(raw_particles, (0,1,3,2)) # [B,T,4,N]
    agent_relative_particles = np.matmul(Vmat, raw_particles)
    agent_relative_particles = np.transpose(agent_relative_particles, (0,1,3,2))
    agent_relative_particles = agent_relative_particles[:,:,:,0:3] # remove homogeneous coordinate

    return agent_relative_particles


def agent_particles_to_image_coordinates(particles, Pmat=None, H_out=256, W_out=256, to_integers=False, eps=1.0e-8):
    '''
    Map agent-relative particles in 3-space to image coordinates in 2-space
    via a perspective projection, encoded in Pmat

    #### INPUTS ####
    ps: [B,T,N,3] agent-relative particle (x,y,z) coordinates
    Pmat: [4,4] matrix that performs perspective projection on homogeneous coordinates
    H_out, W_out: the dimensions of the image to project onto
    to_integers: if True, returned coordinates are typed as integers and can serve as indices

    #### RETURNS ####
    particle_image_coordinates: [B,T,N,2] coordinates in 2-space between [0,H_out] x [0,W_out]
                                By convention the first coordinate indexes the vertical dimension.
    '''
    ### modified from tf into numpy arrays
    shape = list(np.shape(particles))
    if len(shape) != 4:
        particles = np.reshape(particles, (1,1,-1,3))
    shape = list(np.shape(particles))
    if len(list(np.shape(Pmat))) != 4:
        Pmat = np.zeros([shape[0], shape[1], 4, 4], dtype=np.float32) + np.reshape(Pmat, (1,1,4,4))
    ps = np.concatenate([particles, np.ones(shape[0:3]+[1], dtype=np.float32)], axis=3)
    ps = np.transpose(ps, (0,1,3,2))
    ps = np.matmul(Pmat, ps)
    ps = np.transpose(ps, (0,1,3,2))
    # normalize by homogeneous coordinate, then remove it
    ps = np.divide(ps, ps[:,:,:,3:4]+eps)[:,:,:,0:3]

    # map to image coordinates; y axis is inverted by convention
    xims = float(W_out) * 0.5 * (ps[:,:,:,0:1] + 1.0)
    yims = float(H_out) * (1.0 - 0.5*(ps[:,:,:,1:2] + 1.0))

    if to_integers:
        xims = np.floor(xims).astype(dtype=int)
        yims = np.floor(yims).astype(dtype=int)

        xims = np.minimum(np.maximum(xims, 0), W_out-1)
        yims = np.minimum(np.maximum(yims, 0), H_out-1)

    particle_image_coordinates = np.concatenate([yims, xims], axis=3)
    return particle_image_coordinates

def project_and_occlude_particles(particles,
                                  projection_matrix,
                                  im_size=[256,256],
                                  particles_mask=None,
                                  p_radius=0.0,
                                  xyz_dims=[0,3],
                                  particles_agent_centered=False,
                                  camera_matrix=None,
                                  **kwargs
):
    '''
    particles: [B,T,N,D] where the last axis contains both xyz_dims and color_dims (slices)
    im_size: [H,W] ints
    projection_matrix: [B,T,4,4] the matrix for projecting from euclidean space into canonical image coordinates
    xyz_dims: [xdim, zdim+1] must have a difference of 3

    p_radius: float that determines how particles will occlude one another.
    particles_agent_centered: if false, use camera_matrix ("V") to put particles in camera-relative coordinates
    camera_matrix: [B,T,4,4] the matrix "V" for converting global to camera-relative coordinates

    returns
    particles_im_indices: [B,T,N,2] of int32 indices into H and W dimensions of an image of size H,W
    not_occluded_mask: [B,T,N,1] float32 where 1.0 indicates the particle wasn't occluded at radius p_radius, 0.0 otherwise
    '''
    shape = list(np.shape(particles))
    if len(shape) != 4:
        particles = np.reshape(particles, (1,1,-1,3))
    B,T,N,D = list(np.shape(particles))
    H,W = im_size
    if p_radius is None:
        p_radius = 3.0 * (np.minimum(H,W).astype(float) / 256.0)
        print("p radius", p_radius)
    # get indices into image
    particles_xyz = particles[...,xyz_dims[0]:xyz_dims[1]]
    if not particles_agent_centered:
        assert camera_matrix is not None, "need a camera matrix to put into agent coordinates"
        particles_agent = raw_to_agent_particles(particles_xyz, Vmat=camera_matrix)
    else:
        particles_agent = particles_xyz

    # [B,T,N,2] <tf.float32> values in range [0,H]x[0,W]
    particles_depths = particles_agent[...,-1:]
    particles_im_coordinates = agent_particles_to_image_coordinates(particles_agent,
                                                                    Pmat=projection_matrix,
                                                                    H_out=H, W_out=W,
                                                                    to_integers=False)

    return particles_im_coordinates



class Primitive(Dataset):
    def __init__(self, list_path, index_list, sequence_len,
                 normalize_images=True, max_samples=None, training=False):
        super().__init__()

        self.RW = self.RH = self.W = self.H = 256
        self.orig_W = self.orig_H = 256
        self.list_path = list_path  ###note now that this is a path that points to the list of all hdf5 files
        self.max_samples = max_samples
        self.num_obj = 0
        self.training = training
        self.sequence_len = sequence_len


        transform = [Resize((self.H, self.W)), T.ToTensor()]
        obj_transform = [Resize((self.RH, self.RW)), T.ToTensor()]
        if normalize_images:
            transform.append(imagenet_preprocess())
            obj_transform.append(imagenet_preprocess())
        self.transform = T.Compose(transform)
        self.obj_transform = T.Compose(obj_transform)
        self.index_list = index_list
        self.data = self.get_data()

    def get_data(self):
        path = self.list_path
        h5 = h5py.File(path, 'r')
        return h5

    def get_world(self, index):
        '''
        :param index:
        :return: obj_inds: O-len list
                 colors: O-len list

        '''
        world_info = self.data['worldinfo'][index]
        world_info = json.loads(world_info)
        proj_matrix = np.reshape(np.asarray(world_info['projection_matrix']),(4,4))
        cam_matrix = np.reshape(np.asarray(world_info['camera_matrix']),(4,4))
        obj_inds = []
        colors = []
        poss = []
        dicts = world_info['observed_objects']
        num = len(dicts)
        for i in range(num):
            obj_inds.append(dicts[i][1])
            col = np.asarray(dicts[i][24])
            colors.append(col)
            pos = np.asarray(dicts[i][2])
            poss.append(pos)
        poss = np.concatenate(poss, axis = 0) ## (O,3)

        return proj_matrix, cam_matrix, obj_inds, colors, poss

    def get_images(self, index):
        '''
        :param index:
        :return: image of size (H, W, C)
        '''
        img = np.asarray(self.data['images1'][index,:,:,:])
        img = Image.fromarray(img)
        return img

    def get_ids(self, index):
        '''
        :param index:
        :return: ids of size (C, H, W)
        '''
        ids = np.reshape(np.asarray(self.data['objects1'][index, :, :, :]), (self.H, self.W, 3))
        return ids

    def get_projected_positions(self, proj_matrix, cam_matrix, pos, get_int = False):
        '''
        :param proj_matrix: (4,4) projection matrix
        :param cam_matrix: (4,4) camera matrix
        :param pos: positions of particle (O, 3)
        :param get_int: whether to return a integer-values output
        :return:
        '''
        pos = project_and_occlude_particles(pos,
                                      projection_matrix=proj_matrix,
                                      im_size=[256, 256],
                                      particles_mask=None,
                                      p_radius=0.0,
                                      xyz_dims=[0, 3],
                                      particles_agent_centered=False,
                                      camera_matrix=cam_matrix)
        if get_int:
            pos = np.round(pos)

        return pos

    def get_segments(self, ids, colors, pos):
        '''
        index: as usual
        :param ids: numpy array giving ids as images consisting of N colors, where each N corresponds to an object
        :param colors:list of colors the object corresponds to
        :param pos: (O,2) positions of each object center
        :return: get segments of individual objects and concatenate them based on the sequence given in pos of size (O, 1, RH, RW)
        also get bboxes (O,4)
        '''

        #crops = []
        boxes = []
        W, H = self.W, self.H
        for i in range(len(colors)):
            #img = np.zeros([W, H])
            minH = 255
            maxH = 0
            minW = 255
            maxW = 0
            for j in range(H):
                for k in range(W):
                    if np.all(ids[j, k, :] == colors[i]):
                        #img[j, k] = 1
                        minH = min(j, minH)
                        maxH = max(j, maxH)
                        minW = min(k, minW)
                        maxW = max(k, maxW)
            radius = max(pos[i][0] - minH, maxH - pos[i][0], pos[i][1] - minW, maxW -pos[i][1])
            bbox = np.array([pos[i][0], pos[i][1], 2*radius, 2*radius])
            #img = np.reshape(img, (1,1,H,W))
            bbox = np.reshape(bbox, (1,4))
            #crops.append(img)
            boxes.append(bbox)
        ## concatenate
        #segments = np.concatenate(crops, axis = 0)
        bboxes = np.concatenate(boxes, axis = 0)
        return bboxes

    def __len__(self):
        num = len(self.index_list) * self.sequence_len
        if self.max_samples is not None:
            return min(self.max_samples, num)
        return num


    def __getitem__(self, index): #build_one_dataset
        """
        :return: src, dst. each is a list of object
        - 'image': FloatTensor of shape ( , C, H, W). resize and normalize to faster-rcnn
        - 'crop': (O, C, RH, RW) RH >= 224
        - 'bbox': (O, 4) in xyxy (0-1) / xy logw logh
        - 'trip': (T, 3)
        - 'index': (dt,)
        """
        # the following is based on the hdf5 hierarchy
        vid_point = []
        #curr_data_path = os.path.join(self.list_path, '%04d' % index + self.ext)
        l = []
        for i in range(self.sequence_len):
            l.append(self.sequence_len * index + i)

        for i in l:
            vid_obj = {}
            index = i
            images = self.get_images(index)  # Image.fromarray is the last operation
            ids = self.get_ids(index)
            proj_matrix, cam_matrix, obj_inds, colors, poss = self.get_world(index)
            num_obj = len(obj_inds)
            pos = self.get_projected_positions(proj_matrix = proj_matrix, cam_matrix = cam_matrix, pos = poss, get_int=True) #(1,1,O, 2)
            pos = np.reshape(pos, (-1,2)) #(O,2)
            bboxes = self.get_segments(ids, colors, pos) # (O, 4)
            crop = self._crop_image(images, bboxes) #(O, C, H, W)
            images = self.transform(images.convert('RGB')) #(C H, W)
            vid_obj['image'] = images
            vid_obj['crop'] = crop
            vid_obj['bbox'] = torch.FloatTensor(bboxes)
            vid_obj['trip'] = self._build_graph(num_obj)
            valid = np.arange(3, dtype=np.int64)
            vid_obj['info'] = (self.orig_W, self.orig_H, valid)
            vid_obj['valid'] = torch.LongTensor(valid)
            vid_point.append(vid_obj)
        return vid_point


    def _crop_image(self, image, bboxes):
        '''
        :param image: image
        :param bboxes:
        :return: (o, C, H, W)
        '''
        crop_obj = []
        num = np.shape(bboxes)[0]
        for i in range(num):
            s = bboxes[i, :]
            h_c = s[0]
            w_c = s[1]
            r_w = int(s[3]/2)
            r_h = int(s[2]/2)
            left = w_c - r_w
            right = w_c + r_w
            top = h_c - r_h
            bottom = h_c + r_h
            crop = image.crop((left, top, right, bottom)).convert('RGB')
            crop = self.transform(crop)
            crop_obj.append(crop)
        crop_obj = torch.stack(crop_obj)
        return crop_obj


    def _build_graph(self, num_obj):
        all_trip = np.zeros([0, 3], dtype=np.float32)
        for i in range(num_obj):
            for j in range(num_obj):
                trip = [i, 0, j]
                all_trip = np.vstack((all_trip, trip))
        return torch.FloatTensor(all_trip)


def dt_collate_fn(batch):
    """
    :return: src dst. each is a list with dict element
    - 'index': list of str with length N
    - 'image': list of FloatTensor in shape (Dt, V, 1, C, H, W)
    - 'crop': list of FloatTensor in shape (Dt, V, o, C, RH, RW)
    - 'bbox': (Dt, V, o, 4)
    - 'trip': (Dt, V, t, 3)
    """
    key_set = ['index', 'image', 'crop', 'bbox', 'trip', 'valid']
    all_batch = {}
    dt = len(batch[0])
    V = len(batch)
    for key in key_set:
        all_batch[key] = []

    for f in range(dt):
        for v in range(V):
            frame = batch[v][f]
            for key in key_set:
                all_batch[key].append(frame[key])

    for key in all_batch:
        if key == 'index':
            continue
        if key in ['image', 'crop']:
            tensor = torch.stack(all_batch[key])
            all_batch[key] = tensor.view(dt, V, -1, 3, tensor.size(-2), tensor.size(-1))
        elif key in ['bbox', 'trip', 'valid']:
            tensor = torch.stack(all_batch[key])
            all_batch[key] = tensor.view(dt, V, -1, tensor.size(-1))
        else:
            print('key not exist', key)
            raise KeyError

    return all_batch


def build_vid_loaders(args):
    '''args need: n_tr_per_dataset, n_val_per_dataset, dataset_names
    image_dir needs filling in
    '''
    dset_kwargs = {
        'max_samples': None,
        'sequence_len': args.sequence_len,
        #'radius': args.radius,
        'training': args.is_train,
    }
    loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.loader_num_workers,
        'collate_fn': dt_collate_fn,
    }
    list_paths = 'C:/Users/john/Downloads/tdw_prim_sample.hdf5' ## would be 'C:/Users/john/Downloads/tdw_prim_sample.hdf5' on my computer
    data_path = list_paths

    if args.is_train:
        dset_kwargs['list_path'] = data_path
        dset_kwargs['index_list'] = []
        for i in range(int(3072/args.sequence_len)):
            dset_kwargs['index_list'].append(i)
        loader_kwargs['shuffle'] = True
    else:
        print('val')
        dset_kwargs['list_path'] = data_path
        for i in range(int(1024/args.sequence_len)):
            dset_kwargs['index_list'].append(i + int(1024/args.sequence_len))
        dset_kwargs['max_samples'] = args.num_val_samples
        loader_kwargs['shuffle'] = args.shuffle_val
    dset_kwargs['training'] = args.is_train
    train_dset = Primitive(**dset_kwargs)

    loader = DataLoader(train_dset, **loader_kwargs)

    return loader


if __name__ == '__main__':
    # from  import parser_helper
    from cfgs.train_cfgs import TrainOptions
    #from cfgs.test_cfgs import TestOptions
    args = TrainOptions().parse()
    torch.manual_seed(123)
    # args.batch_size=4
    args.sequence_len=16
    train_loader = build_vid_loaders(args)
    key_set = ['index', 'image', 'crop', 'bbox', 'trip']

    for batch in train_loader:
        for key in key_set:
            if key == 'index':
                print(batch[key])
            else:
                pass
                print('size', key, batch[key].size())
        break
