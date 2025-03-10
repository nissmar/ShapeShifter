import argparse
import os
import pymeshlab as ml
import torch
from tqdm import tqdm


def mesh_point_cloud(filepath, depth=9, targetfacenum=10000, text_size=2048, samplespernode=1., pointweight=10, use_poisson=True):
    ms = ml.MeshSet()
    ms.load_new_mesh(filepath)
    if use_poisson:
        ms.apply_filter('generate_surface_reconstruction_screened_poisson',
                        samplespernode=samplespernode, pointweight=pointweight, depth=depth)
    else:
        ms.apply_filter('generate_marching_cubes_apss', resolution=512)

    ms.save_current_mesh('large_mesh.ply')
    ms.apply_filter('meshing_decimation_quadric_edge_collapse',
                    targetfacenum=targetfacenum)
    ms.apply_filter(
        'compute_texcoord_parametrization_triangle_trivial_per_wedge')
    ms.apply_filter('transfer_attributes_to_texture_per_vertex', sourcemesh=0,
                    targetmesh=1, textname="object.png", textw=text_size, texth=text_size)
    ms.save_current_mesh("object.obj", save_vertex_color=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate Model')
    # Experiment
    parser.add_argument('-src',
                        default="output/", type=str, help="mesh name")
    parser.add_argument('-out', default=None, type=str,
                        help="output folder")
    parser.add_argument('-level', default=4,
                        type=int)
    parser.add_argument('-num', default=10,
                        type=int)
    parser.add_argument('-target_face_num', default=10000,
                        type=int)
    parser.add_argument('-use_poisson', default=True,
                        type=bool, help='use poisson or APSS')
    args = parser.parse_args()

    SRC = args.src+'/'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.out is None:
        OUT = SRC.replace("output", "meshed_output")
    else:
        OUT = args.out

    os.mkdir(OUT)
    for name in tqdm(os.listdir(SRC)):
        base_save_path = '{}/{}'.format(OUT, name)

        data_mat = './data/materials/{}.mtl'.format(
            name)
        if not os.path.isfile(data_mat):
            print('Warning: material not found, switching to default mat')
            data_mat = './data/materials/default.mtl'
        os.mkdir(base_save_path)
        for i in range(args.num):
            save_path = '{}/00{}'.format(base_save_path, i)
            os.mkdir(save_path)
            print(save_path)
            filepath = '{}/{}/gen_{}_{}.ply'.format(SRC, name, i, args.level)

            mesh_point_cloud(
                filepath, targetfacenum=args.target_face_num, use_poisson=args.use_poisson)

            os.system('scp {} {}/object.obj.mtl'.format(data_mat, save_path))
            os.system('mv object.obj {}/object.obj'.format(save_path))
            os.system('mv object.png {}/object.png'.format(save_path))
            os.system(
                'mv large_mesh.ply {}/large_mesh.ply'.format(save_path))
    os.system('rm object.obj.mtl')
