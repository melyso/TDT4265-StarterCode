import os
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import seaborn as sns
import pandas as pd

annotations_dir = 'RDD2020_filtered/Annotations/'

widths = []
heights = []
aspect_ratio = []
colors = []
names = []
color_dict = {}

for file_name in os.listdir(annotations_dir):
    tree = ET.parse(annotations_dir+file_name)
    root = tree.getroot()
    print(file_name)
    print(root.tag)
    for child in root:
        if child.tag == 'object':
            name_pos = None
            bnd_box_pos = None
            for pos, grand_children in enumerate(child):
                if grand_children.tag == 'name':
                    name_pos = pos
                if grand_children.tag == 'bndbox':
                    bnd_box_pos = pos
                
            name = child[name_pos].text
            names.append(name)
            x_min, y_min, x_max, y_max = [child[bnd_box_pos][x].text for x in range(4)]
            print(x_min, y_min, x_max, y_max)
            w = int(x_max) - int(x_min)
            h = int(y_max) - int(y_min)
            widths.append(w)
            heights.append(h)
            aspect_ratio.append(w/h)
            colors.append(color_dict.setdefault(name, len(color_dict)))

fig, ax = plt.subplots()
scatter = ax.scatter(widths, heights, 0.1, colors)
legend_handles, legend_labels = scatter.legend_elements()
legend_labels = color_dict.keys()
legend = ax.legend(legend_handles, legend_labels, title='Type of crack')
ax.add_artist(legend)
plt.xlabel('width')
plt.ylabel('height')
plt.show()

df = pd.DataFrame({'names':names, 'aspect_ratio':aspect_ratio})
boxplot = sns.boxplot(x='names', y='aspect_ratio' , data=df)
plt.show()
print(df.groupby('names').describe())
