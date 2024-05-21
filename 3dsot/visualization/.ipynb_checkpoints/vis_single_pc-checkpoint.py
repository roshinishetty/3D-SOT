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
  scene_id = './vis/0020/'

  file_id = '0020-3'
  ind=0         
  print(join(scene_id, file_id))
  dbfile = open(join(scene_id, file_id), 'rb')   
  db = pickle.load(dbfile)
  print("*"*100)
  print(db)
  print("*"*100)
  count = 0
  print(len(db))
      # continue
  ind = 0

  pc_path = f"{db[ind]['anno']['scene']}/{'{:06}.bin'.format(db[ind]['anno']['frame'])}"
  scan_dir = pc_path#f'examples\\kitti\\velodyne\\{file_id}.bin'
  scan = np.fromfile(scan_dir, dtype=np.float32).reshape(-1, 4)
  print(scan.shape)
  fig = mlab.figure(bgcolor=(1, 1, 1), size=(1280, 720))
# plotLine point cloud
  plot = mlab.points3d(scan[:, 0], scan[:, 1], scan[:, 2], mode="point", figure=fig, color=(0,0,0))

  
  # for db_ind in pcloc_db[f"{db[ind]['anno']['scene']}/{'{:06}.bin'.format(db[ind]['anno']['frame'])}"]:
  count +=1
  # print(f"{db[ind]['anno']['scene']}/{'{:06}.bin'.format(db[ind]['anno']['frame'])}")
  
  # print(db[ind])
  
  db_ind = db[ind]
  res = db_ind['results_bb']
  anno = db_ind['anno']
  lab, h, w, l, x, y, z = anno['type'],anno['height'],anno['width'],anno['length'],res.center[0],res.center[1]+anno['height']/2,res.center[2]
  rot = (res.orientation / Quaternion(
                  axis=[1, 0, 0], radians=np.pi / 2)).radians
  # rot = db[0]['results_bb'].orientation.radians
  print(rot)
  h, w, l, x, y, z, rot = map(float, [h, w, l, x, y, z, rot])
  if lab != 'DontCare':
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    corners_3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)



# box["scene"],'{:06}.bin'.format(box["frame"])
# (orientation/Quaternion(axis=[1, 0, 0], radians=np.pi / 2)).radians
      # transform the 3d bbox from object coordiante to camera_0 coordinate
    R = np.array([[np.cos(rot), 0, np.sin(rot)],
                  [0, 1, 0],
                  [-np.sin(rot), 0, np.cos(rot)]])
    corners_3d = np.dot(R, corners_3d).T + np.array([x, y, z])

    # transform the 3d bbox from camera_0 coordinate to velodyne coordinate
    corners_3d = corners_3d[:, [2, 0, 1]] * np.array([[1, -1, -1]])


    def plotLine(p1, p2, front=1):
      mlab.plot3d([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                  color=colors[names.index(lab) * 2 + front], tube_radius=None, line_width=2, figure=fig)


    # plotLine the upper 4 horizontal lines
    plotLine(corners_3d[0], corners_3d[1], 0)  # front = 0 for the front lines
    plotLine(corners_3d[1], corners_3d[2])
    plotLine(corners_3d[2], corners_3d[3])
    plotLine(corners_3d[3], corners_3d[0])

    # plotLine the lower 4 horizontal lines
    plotLine(corners_3d[4], corners_3d[5], 0)
    plotLine(corners_3d[5], corners_3d[6])
    plotLine(corners_3d[6], corners_3d[7])
    plotLine(corners_3d[7], corners_3d[4])

    # plotLine the 4 vertical lines
    plotLine(corners_3d[4], corners_3d[0], 0)
    plotLine(corners_3d[5], corners_3d[1], 0)
    plotLine(corners_3d[6], corners_3d[2])
    plotLine(corners_3d[7], corners_3d[3])

    mlab.view(azimuth=230, distance=50)
    mlab.savefig(filename=f'./images_exp/{file_id}_{ind}.png')
    # mlab.close(all=True)
    mlab.show()
  #       if count==1: time.sleep(5000)
  # mlab.show()



