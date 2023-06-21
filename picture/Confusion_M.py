import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set font size and family
mpl.rcParams['font.size'] =12
mpl.rcParams['font.family'] = 'Times New Roman'

# Load confusion matrix from CSV file
confusion_matrix = np.loadtxt('SESBResNet50_NWPU2confusion_matrix.csv', delimiter=',')
# confusion_matrix = np.loadtxt('../SESBResNet50_AID5confusion.csv', delimiter=',')
# 计算每个类别的样本数量
class_counts = np.sum(confusion_matrix, axis=1)
# 将混淆矩阵的每个元素除以该类别的样本数量，得到该类别的分类准确率
confusion_matrix = (confusion_matrix / class_counts[:, np.newaxis])*100
# Define class labels
# labels = ['Airport', 'BareLand', 'BaseballField', 'Beach', 'Bridge', 'Center', 'Church', 'Commercial', 'DenseResidential', 'Desert', 'Farmland', 'Forest', 'Industrial', 'Meadow', 'MediumResidential', 'Mountain', 'Park', 'Parking', 'Playground', 'Pond', 'Port', 'RailwayStation', 'Resort', 'River', 'School', 'SparseResidential', 'Square', 'Stadium', 'StorageTanks', 'Viaduct']
labels = ['airplane', 'airport', 'baseball_diamond', 'basketball_court', 'beach', 'bridge', 'chaparral', 'church', 'circular_farmland', 'cloud', 'commercial_area', 'dense_residential', 'desert', 'forest', 'freeway', 'golf_course', 'ground_track_field', 'harbor', 'industrial_area', 'intersection', 'island', 'lake', 'meadow', 'medium_residential', 'mobile_home_park', 'mountain', 'overpass', 'palace', 'parking_lot', 'railway', 'railway_station', 'rectangular_farmland', 'river', 'roundabout', 'runway', 'sea_ice', 'ship', 'snowberg', 'sparse_residential', 'stadium', 'storage_tank', 'tennis_court', 'terrace', 'thermal_power_station', 'wetland']
# Set figure size
fig, ax = plt.subplots(figsize=(21,19))

# Plot heatmap with origin set to 'lower'
im = ax.imshow(confusion_matrix, cmap='Blues', extent=[0, len(labels), 0, len(labels)], origin='lower')

# Set ticks at the middle of each cell
ticks = np.arange(len(labels))
plt.xticks(ticks+0.5, labels, rotation=90, fontsize=15)
plt.yticks(ticks+0.5, labels, fontsize=15)
# Hide tick marks
plt.tick_params(axis='both', which='both', length=0)

# Add color bar
cbar = ax.figure.colorbar(im, ax=ax, shrink=0.9)




# Set axis labels and title
plt.xlabel('Predicted Labels', fontsize=17)
plt.ylabel('True Labels', fontsize=17)
plt.title('NWPU(2:8) Confusion Matrix',fontsize=18)



# Add text labels to each cell
for i in range(len(labels)):
    for j in range(len(labels)):
        if confusion_matrix[i, j] != 0:
            color = 'white' if i == j else 'black'
            text = ax.text(j + 0.5, i + 0.5, '{:.1f}'.format(confusion_matrix[i, j]), ha='center', va='center',
                           color=color, fontsize=12)

# Invert y-axis
ax.invert_yaxis()

# Show plot
plt.show()