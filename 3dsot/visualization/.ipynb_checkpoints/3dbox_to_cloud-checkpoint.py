import numpy as np
import seaborn as sns
import mayavi.mlab as mlab
import pickle
from pyquaternion import Quaternion
from os import listdir
from os.path import isfile, join
import time

colors = sns.color_palette('Paired', 9 * 2)
names = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']



if __name__ == '__main__':

  

  scene_id = 'C:/Users/sured/OneDrive/Desktop/674_proj_p2b/simple_kitti_visualization/vis/0020/'
  scene = '0020'
  pcloc_db = {}
  for file_id in listdir(scene_id):
    if isfile(join(scene_id, file_id)):
        dbfile = open(join(scene_id, file_id), 'rb') 
        db = pickle.load(dbfile)
        for ind in range(len(db)):
          print(f"{db[ind]['anno']['scene']}/{'{:06}.bin'.format(db[ind]['anno']['frame'])}")
          if f"{db[ind]['anno']['scene']}/{'{:06}.bin'.format(db[ind]['anno']['frame'])}" in pcloc_db.keys():
            pcloc_db[f"{db[ind]['anno']['scene']}/{'{:06}.bin'.format(db[ind]['anno']['frame'])}"].append(db[ind])
          else:
            pcloc_db[f"{db[ind]['anno']['scene']}/{'{:06}.bin'.format(db[ind]['anno']['frame'])}"] = [db[ind]]

  process_scene = False

  # file_id = '0020-0'
  # ind=0         
  # print(join(scene_id, file_id))
  # dbfile = open(join(scene_id, file_id), 'rb')    
  # db = pickle.load(dbfile)
  if process_scene:
    count = 0
    for frame in range(154):
        # continue
      # ind = 0
      pc_path = f"{scene}/{'{:06}.bin'.format(frame)}"
      scan_dir = pc_path#f'examples\\kitti\\velodyne\\{file_id}.bin'
      scan = np.fromfile(scan_dir, dtype=np.float32).reshape(-1, 4)
      fig = mlab.figure(bgcolor=(0, 0, 0), size=(1280, 720))
      sce = mlab.gcf()
    # plotLine point cloud
      plot = mlab.points3d(scan[:, 0], scan[:, 1], scan[:, 2], mode="point", figure=fig)
      for db_ind in pcloc_db[pc_path]:
        count +=1
        # print(f"{db[ind]['anno']['scene']}/{'{:06}.bin'.format(db[ind]['anno']['frame'])}")
        
        # print(db[ind])
        

        res = db_ind['results_bb']
        anno = db_ind['anno']
        lab, h, w, l, x, y, z = anno['type'],anno['height'],anno['width'],anno['length'],res.center[0],res.center[1]+anno['height']/2,res.center[2]
        rot = (res.orientation / Quaternion(
                        axis=[1, 0, 0], radians=np.pi / 2)).radians
        # rot = db[0]['results_bb'].orientation.radians
        print(rot)
        h, w, l, x, y, z, rot = map(float, [h, w, l, x, y, z, rot])
        if lab != 'DontCare':
          bbox_x = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
          bbox_y = [0, 0, 0, 0, -h, -h, -h, -h]
          bbox_z = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
          bbox_3d = np.vstack([bbox_x, bbox_y, bbox_z])  # (3, 8)



      # box["scene"],'{:06}.bin'.format(box["frame"])
      # (orientation/Quaternion(axis=[1, 0, 0], radians=np.pi / 2)).radians
            # transform the 3d bbox from object coordiante to camera_0 coordinate
          R = np.array([[np.cos(rot), 0, np.sin(rot)],
                        [0, 1, 0],
                        [-np.sin(rot), 0, np.cos(rot)]])
          bbox_3d = np.dot(R, bbox_3d).T + np.array([x, y, z])

          # transform the 3d bbox from camera_0 coordinate to velodyne coordinate
          bbox_3d = bbox_3d[:, [2, 0, 1]] * np.array([[1, -1, -1]])


          def plotLine(p1, p2, front=1):
            mlab.plot3d([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                        color=colors[names.index(lab) * 2 + front], tube_radius=None, line_width=2, figure=fig)


          # plotLine the upper 4 horizontal lines
          plotLine(bbox_3d[0], bbox_3d[1], 0)  # front = 0 for the front lines
          plotLine(bbox_3d[1], bbox_3d[2])
          plotLine(bbox_3d[2], bbox_3d[3])
          plotLine(bbox_3d[3], bbox_3d[0])

          # plotLine the lower 4 horizontal lines
          plotLine(bbox_3d[4], bbox_3d[5], 0)
          plotLine(bbox_3d[5], bbox_3d[6])
          plotLine(bbox_3d[6], bbox_3d[7])
          plotLine(bbox_3d[7], bbox_3d[4])

          # plotLine the 4 vertical lines
          plotLine(bbox_3d[4], bbox_3d[0], 0)
          plotLine(bbox_3d[5], bbox_3d[1], 0)
          plotLine(bbox_3d[6], bbox_3d[2])
          plotLine(bbox_3d[7], bbox_3d[3])
          sce.scene.camera.position = [-33.50590757376398, -16.60769924150611, 11.048512265753493]
          sce.scene.camera.focal_point = [2.4076986557666262, -0.2019878076848256, -1.141613178752122]
          sce.scene.camera.view_angle = 30.0
          sce.scene.camera.view_up = [0.2981163341319765, 0.05545325595212023, 0.9529174086603835]
          sce.scene.camera.clipping_range = [0.235881608077219, 235.881608077219]
          sce.scene.camera.compute_view_plane_normal()
          sce.scene.render()
          # azimuth, elevation, distance, focalpoint = mlab.view()
          # mlab.yaw([0.2868281800114307, -0.06620167400698805, 0.9556918611717917])

      mlab.savefig(filename=f'./images/{scene}_{frame}.png')
      mlab.close(all=True)
      # mlab.show()
      # print(print(mlab.view()))
      # if count>0: time.sleep(5000)
    # mlab.show()
  else:
    file_id = '0020-21'
    gt = False
    ind=0         
    print(join(scene_id, file_id))
    dbfile = open(join(scene_id, file_id), 'rb')    
    db = pickle.load(dbfile)
    count = 0
    for ind in range(len(db)):
        # continue
      # ind = 0
      pc_path = f"{db[ind]['anno']['scene']}/{'{:06}.bin'.format(db[ind]['anno']['frame'])}"
      scan_dir = pc_path#f'examples\\kitti\\velodyne\\{file_id}.bin'
      scan = np.fromfile(scan_dir, dtype=np.float32).reshape(-1, 4)
      fig = mlab.figure(bgcolor=(0, 0, 0), size=(1280, 720))
    # plotLine point cloud
      plot = mlab.points3d(scan[:, 0], scan[:, 1], scan[:, 2], mode="point", figure=fig)

      # ll = np.arange(10)
      # lot = mlab.points3d(scan[:10, 0], scan[:10, 1], scan[:10, 2],ll, mode="point", figure=fig, color=(1.0,0.0,0.0))
      # lot.actor.property.point_size = 20
      # lot.actor.property.render_points_as_spheres =  True
      for db_ind in pcloc_db[f"{db[ind]['anno']['scene']}/{'{:06}.bin'.format(db[ind]['anno']['frame'])}"]:
        count +=1
        # print(f"{db[ind]['anno']['scene']}/{'{:06}.bin'.format(db[ind]['anno']['frame'])}")
        
        # print(db[ind])
        

        res = db_ind['results_bb']
        anno = db_ind['anno']
        lab, h, w, l, x, y, z = anno['type'],anno['height'],anno['width'],anno['length'],res.center[0],res.center[1]+anno['height']/2,res.center[2]
        lab, h, w, l, xo, yo, zo = anno['type'],anno['height'],anno['width'],anno['length'],anno['x'],anno['y'],anno['z']
        roto = anno['rotation_y']
        rot = (res.orientation / Quaternion(
                        axis=[1, 0, 0], radians=np.pi / 2)).radians
        # rot = db[0]['results_bb'].orientation.radians
        print(rot)
        h, w, l, x, y, z, rot = map(float, [h, w, l, x, y, z, rot])
        ho, wo, lo, xo, yo, zo, roto = map(float, [h, w, l, xo, yo, zo, roto])
        if lab != 'DontCare':
          bbox_x = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
          bbox_y = [0, 0, 0, 0, -h, -h, -h, -h]
          bbox_z = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
          bbox_3d = np.vstack([bbox_x, bbox_y, bbox_z])  # (3, 8)
          bbox_3do = np.vstack([bbox_x, bbox_y, bbox_z])  # (3, 8)


      # box["scene"],'{:06}.bin'.format(box["frame"])
      # (orientation/Quaternion(axis=[1, 0, 0], radians=np.pi / 2)).radians
            # transform the 3d bbox from object coordiante to camera_0 coordinate
          R = np.array([[np.cos(rot), 0, np.sin(rot)],
                        [0, 1, 0],
                        [-np.sin(rot), 0, np.cos(rot)]])
          bbox_3d = np.dot(R, bbox_3d).T + np.array([x, y, z])

          # transform the 3d bbox from camera_0 coordinate to velodyne coordinate
          bbox_3d = bbox_3d[:, [2, 0, 1]] * np.array([[1, -1, -1]])

          Ro = np.array([[np.cos(roto), 0, np.sin(roto)],
                        [0, 1, 0],
                        [-np.sin(roto), 0, np.cos(roto)]])
          bbox_3do = np.dot(Ro, bbox_3do).T + np.array([xo, yo, zo])

          # transform the 3d bbox from camera_0 coordinate to velodyne coordinate
          bbox_3do = bbox_3do[:, [2, 0, 1]] * np.array([[1, -1, -1]])


          def plotLine(p1, p2, front=1, add=0):
            if not add:
                mlab.plot3d([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                        color=(colors[names.index(lab) * 2 + front]), tube_radius=None, line_width=2, figure=fig)
            else:
              mlab.plot3d([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                        color=(1,0,0), tube_radius=None, line_width=2, figure=fig)


          # plotLine the upper 4 horizontal lines
          plotLine(bbox_3d[0], bbox_3d[1], 0)  # front = 0 for the front lines
          plotLine(bbox_3d[1], bbox_3d[2])
          plotLine(bbox_3d[2], bbox_3d[3])
          plotLine(bbox_3d[3], bbox_3d[0])

          # plotLine the lower 4 horizontal lines
          plotLine(bbox_3d[4], bbox_3d[5], 0)
          plotLine(bbox_3d[5], bbox_3d[6])
          plotLine(bbox_3d[6], bbox_3d[7])
          plotLine(bbox_3d[7], bbox_3d[4])

          # plotLine the 4 vertical lines
          plotLine(bbox_3d[4], bbox_3d[0], 0)
          plotLine(bbox_3d[5], bbox_3d[1], 0)
          plotLine(bbox_3d[6], bbox_3d[2])
          plotLine(bbox_3d[7], bbox_3d[3])

          if gt:

            plotLine(bbox_3do[0], bbox_3do[1], 0,add=1)  # front = 0 for the front lines
            plotLine(bbox_3do[1], bbox_3do[2],add=1)
            plotLine(bbox_3do[2], bbox_3do[3],add=1)
            plotLine(bbox_3do[3], bbox_3do[0],add=1)

            # plotLine the lower 4 horizontal lines
            plotLine(bbox_3do[4], bbox_3do[5], 0,add=1)
            plotLine(bbox_3do[5], bbox_3do[6],add=1)
            plotLine(bbox_3do[6], bbox_3do[7],add=1)
            plotLine(bbox_3do[7], bbox_3do[4],add=1)

            # plotLine the 4 vertical lines
            plotLine(bbox_3do[4], bbox_3do[0], 0,add=1)
            plotLine(bbox_3do[5], bbox_3do[1], 0,add=1)
            plotLine(bbox_3do[6], bbox_3do[2],add=1)
            plotLine(bbox_3do[7], bbox_3do[3],add=1)

          mlab.view(azimuth=230, distance=50)
      mlab.savefig(filename=f'./images_exp/{file_id}_{ind}.png')
      # mlab.close(all=True)
      mlab.show()
  #       if count==1: time.sleep(5000)
  # mlab.show()
