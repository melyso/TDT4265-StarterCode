import os
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import seaborn as sns
import pandas as pd
import numpy as np
import cv2
import json

annotations_dir = 'RDD2020_filtered/Annotations/'

widths = []
heights = []
aspect_ratio = []
colors = []
names = []
color_dict = {}
heat_maps = {}

heat_w, heat_h = 500, 500

def analyze_tdt():
    img_w, img_h = [1920, 1080]
    scale_w = heat_w / img_w
    scale_h = heat_h / img_h

    with open('tdt4265/labels.json') as f:
        labels = json.load(f)
    #print(labels.keys())
    #print(labels['annotations'][0])
    #print(labels['info'])
    #print(labels['licences'])
    #print(labels['categories'])

    nameslist =['D00', 'D10', 'D20', 'D40']
    for annotation in labels['annotations']:

        name = nameslist[ int(annotation['category_id']) ]
        names.append(name)
        colors.append(color_dict.setdefault(name, len(color_dict)))
        heat_map = heat_maps.setdefault(name, np.zeros((heat_w, heat_h)))

        x_top_left, y_top_left, width, height = annotation['bbox']

        widths.append(width)
        heights.append(height)
        aspect_ratio.append(width/height)

        y_min = y_top_left
        y_max = y_top_left + height
        x_min = x_top_left
        x_max = x_top_left + width

        heat_map[int(round(y_min*scale_h)):int(round(y_max*scale_h)), int(round(x_min*scale_w)):int(round(x_max*scale_w))]+=1


def analyze_rdd():
    for file_name in os.listdir(annotations_dir):
        img_w, img_h = None, None
        scale_w, scale_h = None, None
        tree = ET.parse(annotations_dir+file_name)
        root = tree.getroot()
        print(file_name)
        print(root.tag)
        for child in root:
            if child.tag == 'size':
                for pos, grand_child in enumerate(child):
                    if grand_child.tag == 'width':
                        img_w = int(grand_child.text)
                        scale_w = heat_w/img_w
                    if grand_child.tag == 'height':
                        img_h = int(grand_child.text)
                        scale_h = heat_h/img_h
            if child.tag == 'object':
                name_pos = None
                bnd_box_pos = None
                for pos, grand_child in enumerate(child):
                    if grand_child.tag == 'name':
                        name_pos = pos
                    if grand_child.tag == 'bndbox':
                        bnd_box_pos = pos

                name = child[name_pos].text
                names.append(name)
                x_min, y_min, x_max, y_max = [int(child[bnd_box_pos][x].text) for x in range(4)]
                w = x_max - x_min
                h = y_max - y_min

                widths.append(w)
                heights.append(h)
                aspect_ratio.append(w/h)
                colors.append(color_dict.setdefault(name, len(color_dict)))

                heat_map = heat_maps.setdefault(name, np.zeros((heat_w, heat_h)))
                heat_map[int(round(y_min*scale_h)):int(round(y_max*scale_h)), int(round(x_min*scale_w)):int(round(x_max*scale_w))]+=1

analyze_tdt()
#analyze_rdd()

# Display scatterplot of height-width of bounding-boxes.
fig, ax = plt.subplots()
scatter = ax.scatter(widths, heights, 0.1, colors)
legend_handles, legend_labels = scatter.legend_elements()
legend_labels = color_dict.keys()
legend = ax.legend(legend_handles, legend_labels, title='Type of crack')
ax.add_artist(legend)
plt.xlabel('width')
plt.ylabel('height')
plt.show()

# Display boxplot of aspect ratios for different classes.
df = pd.DataFrame({'names':names, 'aspect_ratio':aspect_ratio})
boxplot = sns.boxplot(x='names', y='aspect_ratio' , data=df)
plt.show()
print(df.groupby('names').describe())

# Display heatmap for different classes.
for damage_type in heat_maps.keys():
    plt.imshow(heat_maps[damage_type], cmap='hot', interpolation='nearest')
    plt.title(damage_type)
    plt.show()

# Display combined heatmap.
combined = heat_maps['D00']+heat_maps['D10']+heat_maps['D20']+heat_maps['D40']
plt.imshow(combined, cmap='hot', interpolation='nearest')
plt.title('Combined')
plt.show()

# Display a sample image with combined heatmap as overlay.
img = cv2.imread('RDD2020_filtered/JPEGImages/Japan_005338.jpg', 1)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
combined_resized = cv2.resize(combined, dsize=img.shape[:2], interpolation=cv2.INTER_CUBIC)
combined_resized = np.uint8(combined_resized/np.max(combined_resized)*255)
heatmap_img = cv2.applyColorMap(combined_resized, cv2.COLORMAP_JET)
cv2.imshow('Heatmap from view', cv2.addWeighted(heatmap_img, 0.5, img, 0.5, 0))
cv2.waitKey(0)
