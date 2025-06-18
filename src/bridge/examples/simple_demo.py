import argparse
import os
import cv2

import genesis as gs
from genesis.options.renderers import BatchRenderer
import numpy as np
from genesis.utils.geom import trans_to_T
from genesis.utils.image_exporter import FrameImageExporter

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(backend=gs.cpu if args.cpu else gs.gpu)

    ########################## create a scene ##########################
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        show_viewer=args.vis,
        rigid_options=gs.options.RigidOptions(
            # constraint_solver=gs.constraint_solver.Newton,
        ),
        renderer = gs.options.renderers.BatchRenderer(
            use_rasterizer=True,
        )
    )

    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
        visualize_contact=True,
    )

    ########################## cameras ##########################
    cam_0 = scene.add_camera(
        res=(512, 512),
        pos=(1.5, 0.5, 1.5),
        lookat=(0.0, 0.0, 0.5),
        fov=45,
        GUI=True,
    )
    cam_0.attach(franka.links[6], trans_to_T(np.array([0.0, 0.5, 0.0])))
    cam_1 = scene.add_camera(
        res=(512, 512),
        pos=(3.5, 0.0, 2.5),
        lookat=(0, 0, 0.5),
        fov=30,
        GUI=True,
    )
    scene.add_light(
        pos=[0.0, 0.0, 1.5],
        dir=[1.0, 1.0, -2.0],
        directional=1,
        castshadow=1,
        cutoff=45.0,
        intensity=0.5
    )
    scene.add_light(
        pos=[4, -4, 4],
        dir=[-1, 1, -1],
        directional=0,
        castshadow=1,
        cutoff=45.0,
        intensity=0.5
    )
    ########################## build ##########################
    n_envs = 3
    n_steps = 2
    do_batch_dump = True
    scene.build(n_envs=n_envs)

    # warmup
    scene.step()
    rgb, depth, _, _ = scene.render_all_cams()

    # Create an image exporter
    output_dir = 'img_output/test'
    exporter = FrameImageExporter(output_dir)

    # timer
    from time import time
    start_time = time()

    for i in range(n_steps):
        scene.step()
        if do_batch_dump:
            rgb, depth, normal, _ = scene.render_all_cams()
            exporter.export_frame_all_cams(i, rgb=rgb, depth=depth)
        else:
            rgb, depth, _, _ = cam_0.render()
            exporter.export_frame_single_cam(i, cam_0.idx, rgb=rgb, depth=depth)
    
    end_time = time()
    print(f'n_envs: {n_envs}')
    print(f'Time taken: {end_time - start_time} seconds')
    print(f'Time taken per env: {(end_time - start_time) / n_envs} seconds')
    print(f'FPS: {n_envs * n_steps / (end_time - start_time)}')
    print(f'FPS per env: {n_steps / (end_time - start_time)}')


if __name__ == "__main__":
    main()
