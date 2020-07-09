import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
import random
import cv2
from tensorflow import keras

class Generate_Shapes:
    def __init__(self, dest_dir, sample_size=5000):

        self.sample_size = sample_size
        self.img_size = 200.0
        self.dpi = 100
        self.canvas_size = self.img_size/self.dpi
        self.dest_dir = dest_dir

        self.shapes = ['Rectangle', 'Circle',
                       'Triangle', 'Pentagon', 'Hexagon']

        self.fig = plt.figure(frameon=False)

        self.gen_flag = 0

        self.bbox_prev = []

        os.makedirs(os.path.join(self.dest_dir,
                                 'Shapes_Dataset'), exist_ok=True)

    def iou_calc(self, bbox1, bbox2):
        Ax1, Ay1, Ax2, Ay2 = bbox1[0][0], bbox1[0][1], bbox1[1][0], bbox1[1][1]
        Bx1, By1, Bx2, By2 = bbox2[0][0], bbox2[0][1], bbox2[1][0], bbox2[1][1]

        #inter rect
        Ix1 = max(Ax1, Bx1)
        Iy1 = max(Ay1, By1)
        Ix2 = min(Ax2, Bx2)
        Iy2 = min(Ay2, By2)

        inter_area = max(0, (Ix2-Ix1)*(Iy2-Iy1))

        area_rect1 = (Ax2-Ax1)*(Ay2-Ay1)
        area_rect2 = (Bx2-Bx1)*(By2-By1)

        union_area = area_rect1 + area_rect2 - inter_area

        return inter_area/union_area

    def rect_coords(self, X1, Y1, width, height, angle):
        theta = angle*(np.pi/180.)
        alpha = np.arctan(height/width)
        hypot = np.sqrt(width**2 + height**2)

        x1, y1 = X1, Y1
        x2, y2 = x1+width*np.cos(theta), y1+width*np.sin(theta)
        x3, y3 = x1+hypot*np.cos(theta+alpha), y1+hypot*np.sin(theta+alpha)
        x4, y4 = x1-height*np.sin(theta), y1+height*np.cos(theta)

        rect_coords = [[x1, y1], [x2, y2], [x3, y3], [x4, y4], [x1, y1]]
        rect_coords = np.array(rect_coords)
        r_bbox = [[min(rect_coords[:, 0]), min(rect_coords[:, 1])], [
            max(rect_coords[:, 0]), max(rect_coords[:, 1])]]

        return rect_coords, np.array(r_bbox)

    def triangle_coords(self, X1, Y1, base, side_len, alpha, theta):
        alpha *= (np.pi/180.)
        theta *= (np.pi/180.)

        x1, y1 = X1, Y1
        x2, y2 = x1 + side_len * \
            np.cos(theta+alpha), y1 + side_len*np.sin(theta+alpha)
        x3, y3 = x1 + base*np.cos(theta), y1 + base*np.sin(theta)

        triangle_coords = [[x1, y1], [x2, y2], [x3, y3], [x1, y1]]
        triangle_coords = np.array(triangle_coords)
        t_bbox = [[min(triangle_coords[:, 0]), min(triangle_coords[:, 1])], [
            max(triangle_coords[:, 0]), max(triangle_coords[:, 1])]]

        return triangle_coords, np.array(t_bbox)

    def polygon_coords(self, X0, Y0, num_sides, r, theta):
        alpha = (2*np.pi)/float(num_sides)
        theta = theta*(np.pi/180.)
        poly_coords = []
        for i in range(num_sides):
            x = X0 + r*np.cos(theta+i*alpha)
            y = Y0 + r*np.sin(theta+i*alpha)
            poly_coords.append([x, y])

        poly_coords.append([poly_coords[0][0], poly_coords[0][1]])
        poly_coords = np.array(poly_coords)
        p_bbox = [[min(poly_coords[:, 0]), min(poly_coords[:, 1])], [
            max(poly_coords[:, 0]), max(poly_coords[:, 1])]]

        return poly_coords, np.array(p_bbox)

    def draw_polygon(self, polygon, filename, mask_filename=None):
        # self.fig = plt.figure(frameon=False)
        self.fig.set_size_inches(self.canvas_size, self.canvas_size)
        ax = plt.Axes(self.fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        ax.add_patch(Polygon(polygon, fill=True, fc='k'))
        ax.set_xlim(0, self.canvas_size)
        ax.set_ylim(0, self.canvas_size)
        self.fig.add_axes(ax)
        if self.gen_flag == 1:
            img_filename = filename
            self.fig.savefig(img_filename, dpi=self.dpi)
            plt.close()

        if self.mask_flag:
            fig2 = plt.figure(frameon=False)
            fig2.set_size_inches(self.canvas_size, self.canvas_size)
            ax = plt.Axes(fig2, [0., 0., 1., 1.])
            ax.set_axis_off()
            ax.add_patch(Polygon(polygon, fill=True, fc='k'))
            ax.set_xlim(0, self.canvas_size)
            ax.set_ylim(0, self.canvas_size)
            fig2.add_axes(ax)
            fig2.savefig(mask_filename, dpi=self.dpi)
            plt.close(fig2)

        return

    def draw_circle(self, x1, y1, radius, filename, mask_filename=None):
        # self.fig = plt.figure(frameon=False)
        self.fig.set_size_inches(self.canvas_size, self.canvas_size)
        ax = plt.Axes(self.fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        ax.add_patch(Circle((x1, y1), radius, fill=True, fc='k'))
        ax.set_xlim(0, self.canvas_size)
        ax.set_ylim(0, self.canvas_size)
        self.fig.add_axes(ax)
        if self.gen_flag == 1:
            img_filename = filename
            self.fig.savefig(img_filename, dpi=self.dpi)
            plt.close()

        if self.mask_flag:
            fig2 = plt.figure(frameon=False)
            fig2.set_size_inches(self.canvas_size, self.canvas_size)
            ax = plt.Axes(fig2, [0., 0., 1., 1.])
            ax.set_axis_off()
            ax.add_patch(Circle((x1, y1), radius, fill=True, fc='k'))
            ax.set_xlim(0, self.canvas_size)
            ax.set_ylim(0, self.canvas_size)
            fig2.add_axes(ax)
            fig2.savefig(mask_filename, dpi=self.dpi)
            plt.close(fig2)

        return

    def write_label_file(self, bbox, shape_id, filename):
        lab_filename = filename
        with open(lab_filename, 'a') as f:
            for j in range(2):
                f.write('{}, {},'.format(
                    str(bbox[j, 0]/self.canvas_size), str(bbox[j, 1]/self.canvas_size)))

            f.write('{}\n'.format(str(shape_id)))

        return

    def create_rectangles(self, img_path, lab_path, mask_filename = None):
        min_size = self.canvas_size/10.
        xpad = 0.5
        ypad = 0.5

        for i in range(self.sample_size):

            repeat_flag = True
            while repeat_flag:

                x1, y1 = random.randint(int(xpad*100), int((self.canvas_size-xpad)*100)) / \
                    100, random.randint(
                        int(ypad*100), int((self.canvas_size-ypad)*100))/100

                min_dim = min(x1, y1, self.canvas_size-x1, self.canvas_size-y1)

                if min_size < (min_dim/np.sqrt(2)):
                    width = random.randint(
                        int(min_size*100), int((min_dim/np.sqrt(2))*100))/100
                    height = np.sqrt(min_dim**2 - width**2)
                else:
                    width = min_size
                    height = min_size

                angle = random.randint(0, 90)

                rect, bbox = self.rect_coords(x1, y1, width, height, angle)

                iou_score = []
                if self.bbox_prev:
                    for box in self.bbox_prev:
                        iou_score.append(self.iou_calc(box, bbox))

                # Check if IOU of present BBox and any previous BBox is more than 5%, if so repeat random origin generation
                if any([score>0.05 for score in iou_score]):
                    repeat_flag = True
                else:
                    repeat_flag = False


                if self.gen_flag == 1:
                    self.fig = plt.figure(frameon=False)
                    
                    img_filename = os.path.join(img_path, 'Rectangle_'+str(i)+'.png')
                    lab_filename = os.path.join(lab_path, 'Rectangle_'+str(i)+'.txt')

                    repeat_flag = False
                else:
                    img_filename = img_path
                    lab_filename = lab_path

                    if not repeat_flag:
                        self.bbox_prev.append(bbox)

                if not repeat_flag:
                    self.draw_polygon(rect, img_filename, mask_filename)
                    self.write_label_file(
                        bbox, self.shapes.index('Rectangle'), lab_filename)

            if self.gen_flag == 2:
                break

    def create_triangles(self, img_path, lab_path, mask_filename = None):
        min_size = self.canvas_size/5.
        xpad = 0.5
        ypad = 0.5

        for i in range(self.sample_size):

            repeat_flag = True

            while repeat_flag:

                x1, y1 = random.randint(int(xpad*100), int((self.canvas_size-xpad)*100)) / \
                    100, random.randint(
                        int(ypad*100), int((self.canvas_size-ypad)*100))/100

                min_dim = min(x1, y1, self.canvas_size-x1, self.canvas_size-y1)

                angle = random.randint(0, 90)
                alpha = random.randint(20, 160)

                if min_size < (min_dim):
                    base = random.randint(int(min_size*100), int(min_dim*100))/100
                    side_len = random.randint(
                        int(min_size*100), int(min_dim*100))/100
                else:
                    base = min_size
                    side_len = min_size

                triangle, bbox = self.triangle_coords(
                    x1, y1, base, side_len, alpha, angle)

                iou_score = []
                if self.bbox_prev:
                    for box in self.bbox_prev:
                        iou_score.append(self.iou_calc(box, bbox))

                if any([score>0.05 for score in iou_score]):
                    repeat_flag = True
                else:
                    repeat_flag = False


                if self.gen_flag == 1:
                    self.fig = plt.figure(frameon=False)
                    
                    img_filename = os.path.join(img_path, 'Triangle_'+str(i)+'.png')
                    lab_filename = os.path.join(lab_path, 'Triangle_'+str(i)+'.txt')

                    repeat_flag = False
                else:
                    img_filename = img_path
                    lab_filename = lab_path

                    if not repeat_flag:
                        self.bbox_prev.append(bbox)

                if not repeat_flag:
                    self.draw_polygon(triangle, img_filename, mask_filename)
                    self.write_label_file(
                        bbox, self.shapes.index('Triangle'), lab_filename)

            if self.gen_flag == 2:
                break

    def create_circles(self, img_path, lab_path, mask_filename = None):
        min_size = self.canvas_size/5.
        xpad = 0.5
        ypad = 0.5

        for i in range(self.sample_size):
            repeat_flag = True
            while repeat_flag:
                x1, y1 = random.randint(int(xpad*100), int((self.canvas_size-xpad)*100)) / \
                    100, random.randint(
                        int(ypad*100), int((self.canvas_size-ypad)*100))/100

                radius = min(x1, y1, self.canvas_size-x1, self.canvas_size-y1)/2.

                bbox = np.array([[x1-radius, y1-radius], [x1+radius, y1+radius]])

                iou_score = []
                if self.bbox_prev:
                    for box in self.bbox_prev:
                        iou_score.append(self.iou_calc(box, bbox))

                if any([score>0.05 for score in iou_score]):
                    repeat_flag = True
                else:
                    repeat_flag = False

                if self.gen_flag == 1:
                    self.fig = plt.figure(frameon=False)

                    img_filename = os.path.join(img_path, 'Circle_'+str(i)+'.png')
                    lab_filename = os.path.join(lab_path, 'Circle_'+str(i)+'.txt')

                    repeat_flag = False
                else:
                    img_filename = img_path
                    lab_filename = lab_path

                    if not repeat_flag:
                        self.bbox_prev.append(bbox)

                if not repeat_flag:
                    self.draw_circle(x1, y1, radius, img_filename, mask_filename)
                    self.write_label_file(
                        bbox, self.shapes.index('Circle'), lab_filename)

            if self.gen_flag == 2:
                break

    def create_pentagons(self, img_path, lab_path, mask_filename = None):
        min_size = self.canvas_size/5.
        xpad = 0.5
        ypad = 0.5

        for i in range(self.sample_size):
            repeat_flag = True
            while repeat_flag:
                x1, y1 = random.randint(int(xpad*100), int((self.canvas_size-xpad)*100))/100, random.randint(int(ypad*100), int((self.canvas_size-ypad)*100))/100

                radius = min(x1, y1, self.canvas_size-x1, self.canvas_size-y1)/2.
                angle = random.randint(0, 90)

                pentagon, bbox = self.polygon_coords(x1, y1, 5, radius, angle)

                iou_score = []
                if self.bbox_prev:
                    for box in self.bbox_prev:
                        iou_score.append(self.iou_calc(box, bbox))

                if any([score>0.05 for score in iou_score]):
                    repeat_flag = True
                else:
                    repeat_flag = False

                if self.gen_flag == 1:
                    self.fig = plt.figure(frameon=False)
                    
                    img_filename = os.path.join(img_path, 'Pentagon_'+str(i)+'.png')
                    lab_filename = os.path.join(lab_path, 'Pentagon_'+str(i)+'.txt')

                    repeat_flag = False
                else:
                    img_filename = img_path
                    lab_filename = lab_path

                    if not repeat_flag:
                        self.bbox_prev.append(bbox)

                if not repeat_flag:
                    self.draw_polygon(pentagon, img_filename, mask_filename)
                    self.write_label_file(
                        bbox, self.shapes.index('Pentagon'), lab_filename)

            if self.gen_flag == 2:
                break


    def create_hexagons(self, img_path, lab_path, mask_filename = None):
        min_size = self.canvas_size/5.
        xpad = 0.5
        ypad = 0.5

        for i in range(self.sample_size):
            repeat_flag = True
            while repeat_flag:
                x1, y1 = random.randint(int(xpad*100), int((self.canvas_size-xpad)*100)) / \
                    100, random.randint(
                        int(ypad*100), int((self.canvas_size-ypad)*100))/100

                radius = min(x1, y1, self.canvas_size-x1, self.canvas_size-y1)/2.
                angle = random.randint(0, 90)

                hexagon, bbox = self.polygon_coords(x1, y1, 6, radius, angle)

                iou_score = []
                if self.bbox_prev:
                    for box in self.bbox_prev:
                        iou_score.append(self.iou_calc(box, bbox))

                if any([score>0.05 for score in iou_score]):
                    repeat_flag = True
                else:
                    repeat_flag = False

                if self.gen_flag == 1:
                    self.fig = plt.figure(frameon=False)
                    
                    img_filename = os.path.join(img_path, 'Hexagon_'+str(i)+'.png')
                    lab_filename = os.path.join(lab_path, 'Hexagon_'+str(i)+'.txt')

                    repeat_flag = False
                else:
                    img_filename = img_path
                    lab_filename = lab_path

                    if not repeat_flag:
                        self.bbox_prev.append(bbox)



                if not repeat_flag:
                    self.draw_polygon(hexagon, img_filename, mask_filename)
                    self.write_label_file(
                        bbox, self.shapes.index('Hexagon'), lab_filename)

            if self.gen_flag == 2:
                break


    def gen_data(self):

        self.gen_flag = 1

        for shape in self.shapes:
            img_path = os.path.join(
                self.dest_dir, 'Shapes_Dataset', shape, 'Images')
            lab_path = os.path.join(
                self.dest_dir, 'Shapes_Dataset', shape, 'Labels')
            os.makedirs(img_path, exist_ok=True)
            os.makedirs(lab_path, exist_ok=True)

            if shape == 'Rectangle':
                self.create_rectangles(img_path, lab_path)
            elif shape == 'Triangle':
                self.create_triangles(img_path, lab_path)
            elif shape == 'Circle':
                self.create_circles(img_path, lab_path)
            elif shape == 'Pentagon':
                self.create_pentagons(img_path, lab_path)
            elif shape == 'Hexagon':
                self.create_hexagons(img_path, lab_path)


class Generate_Mixed_Shapes(Generate_Shapes):
    def __init__(self, dest_dir, sample_size=5000, mask_flag = False, grid_size = 5):
        super().__init__(dest_dir=dest_dir, sample_size=sample_size)
        
        self.max_shapes = 3
        self.mask_flag = mask_flag
        self.GRID = grid_size

    def gen_multiple_shapes(self):
        '''
        Generates multiple shapes (max) in a single image file
        '''

        self.gen_flag = 2

        for sample in (range(self.sample_size)):
            print('{}/{}'.format(sample, self.sample_size))
            val = random.randint(1, self.max_shapes)
            self.fig = plt.figure(frameon=False)

            self.bbox_prev = []

            for j in range(val):

                shape = random.choice(self.shapes)

                img_path = os.path.join(
                    self.dest_dir, 'Shapes_Dataset', 'Images')
                lab_path = os.path.join(
                    self.dest_dir, 'Shapes_Dataset', 'Labels')
                
                if self.mask_flag:
                    mask_path = os.path.join(
                        self.dest_dir, 'Shapes_Dataset', 'Masks')

                os.makedirs(img_path, exist_ok=True)
                os.makedirs(lab_path, exist_ok=True)
                os.makedirs(mask_path, exist_ok=True)

                img_filename = os.path.join(img_path, 'Img_'+str(sample)+'.png')
                lab_filename = os.path.join(lab_path, 'Lab_'+str(sample)+'.txt')

                # print(mask_filename)

                if shape == 'Rectangle':
                    if self.mask_flag:
                        mask_filename = os.path.join(mask_path, 'Mask_'+str(sample)+'_'+str(j)+'_Rectangle'+'.png')
                    self.create_rectangles(img_filename, lab_filename, mask_filename)
                elif shape == 'Triangle':
                    if self.mask_flag:
                        mask_filename = os.path.join(mask_path, 'Mask_'+str(sample)+'_'+str(j)+'_Triangle'+'.png')
                    self.create_triangles(img_filename, lab_filename, mask_filename)
                elif shape == 'Circle':
                    if self.mask_flag:
                        mask_filename = os.path.join(mask_path, 'Mask_'+str(sample)+'_'+str(j)+'_Circle'+'.png')
                    self.create_circles(img_filename, lab_filename, mask_filename)
                elif shape == 'Pentagon':
                    if self.mask_flag:
                        mask_filename = os.path.join(mask_path, 'Mask_'+str(sample)+'_'+str(j)+'_Pentagon'+'.png')
                    self.create_pentagons(img_filename, lab_filename, mask_filename)
                elif shape == 'Hexagon':
                    if self.mask_flag:
                        mask_filename = os.path.join(mask_path, 'Mask_'+str(sample)+'_'+str(j)+'_Hexagon'+'.png')
                    self.create_hexagons(img_filename, lab_filename, mask_filename)

            if (j+1) < self.max_shapes:
                for k in range(j+1, self.max_shapes):
                    mask_filename = os.path.join(mask_path, 'Mask_'+str(sample)+'_'+str(k)+'_Empty'+'.png')
                    array = np.ones((200,200,3))
                    matplotlib.image.imsave(mask_filename, array)


            self.fig.savefig(img_filename, dpi=self.dpi)
            plt.close()

    def generate_yolo_labels(self):
        '''
        Generates Labels for using in YOLO architecture.
        Output is a 5x5 Layer with (no. of shapes + 5) filters
        '''
        dest_folder = os.path.join(
                    self.dest_dir, 'Shapes_Dataset', 'Labels_yolo')

        lab_path = os.path.join(
                    self.dest_dir, 'Shapes_Dataset', 'Labels')

        label_files = os.listdir(lab_path)

        os.makedirs(dest_folder, exist_ok=True)

        for file in label_files:
            coords = np.zeros((3,5))
            coord_temp = np.genfromtxt(os.path.join(lab_path, file), delimiter = ',')
            if len(coord_temp)>3:
                coord_temp = coord_temp.reshape((1, 5))
            else:
                coord_temp = coord_temp.reshape((len(coord_temp), 5))
                                                
            for ind in range(len(coord_temp)):
                coords[ind, :] = coord_temp[ind, :]

            grid_no = []

            width = np.zeros((3,1))
            height = np.zeros((3,1))
            x1 = np.zeros((3,1))
            y1 = np.zeros((3,1))
            cat = np.zeros((3,1))

            for ind, coord in enumerate(coords):
                width[ind] = coord[2]-coord[0]
                height[ind] = coord[3]-coord[1]
                x1[ind] = coord[0]+width[ind]*0.5
                y1[ind] = coord[1]+height[ind]*0.5
                cat[ind] = coord[-1]
                if width[ind]!= 0:
                    grid_nox = (np.ceil(x1[ind]/(1/self.GRID)))
                    grid_noy = (np.ceil(y1[ind]/(1/self.GRID)))
                    grid_no.append(self.GRID*grid_nox + grid_noy)

            y = np.zeros((self.GRID*self.GRID,5+len(self.shapes)+1))    
            for grid in range(self.GRID*self.GRID):
                if grid in grid_no:
                    y[grid, 0] = 1
                    ind = grid_no.index(grid)
                    y[grid, 1] = x1[ind]
                    y[grid, 2] = y1[ind]
                    y[grid, 3] = width[ind]
                    y[grid, 4] = height[ind]
                    y[grid, 5:] = keras.utils.to_categorical(cat[ind]+1, len(self.shapes)+1)[0:]
                    
            np.savetxt(os.path.join(dest_folder, file), y, delimiter = ',', fmt='%f')



if __name__ == '__main__':
    dest_folder = 'Input/Destination/Folder'

	''' Individual Shapes Dataset'''
    # S = Generate_Shapes(dest_dir=dest_folder, sample_size=5000)
    # S.gen_data()

	''' Multiple Shapes Dataset '''
    S = Generate_Mixed_Shapes(dest_dir=dest_folder, sample_size=5000, mask_flag = True)
    S.gen_multiple_shapes()
    S.generate_yolo_labels()


    print('Completed!')

