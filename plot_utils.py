import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

'''
Maps 40 classes to 13 classes for Segmentation
'''
def map_40_to_13(mask):
    mapping = {0:11, 1:4, 2:5, 3:0, 4:3, 5:8, 6:9, 7:11, 8:12,	9:5, 10:7, 11:5, 12:12, 13:9, 14:5, 15:12,	
               16:5, 17:6, 18:6, 19:4, 20:6, 21:2, 22:1, 23:5, 24:10, 25:6, 26:6, 27:6, 28:6, 29:6,	30:6, 
               31:5, 32:6, 33:6, 34:6, 35:6, 36:6, 37:6, 38:5, 39:6, 255:255}
    
    mask = mask.squeeze().numpy().astype(int)

    for r, c in np.ndindex(mask.shape):
        mask[r, c] = mapping[mask[r, c]]

    return torch.tensor(mask, dtype=torch.long).unsqueeze(0)


'''
Creates a triple of image, GT mask and segmented mask and saves it to directory figures
'''
def visualize_img_label(image, gt_label, pr_label, filename='test.png'):
    # --------------------------------------------------------------------------------------------
    # place subplots
    # --------------------------------------------------------------------------------------------
    plt.figure(figsize=(16, 5))

    # Leave everything as it is!!!
    # If then only adjust the wspace value!!!
    plt.subplots_adjust(left=0.005,
                        bottom=0,
                        right=0.84,
                        top=1,
                        wspace=0.01,
                        hspace=0.0)
    
    # --------------------------------------------------------------------------------------------
    # Image
    # --------------------------------------------------------------------------------------------
    # Convert Image from Tensor to Image
    image = image.permute(1, 2, 0).numpy()
    plt.subplot(1, 3, 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image)

    # --------------------------------------------------------------------------------------------
    # Label Preprocessing
    # --------------------------------------------------------------------------------------------
    gt_label = map_40_to_13(gt_label)
    pr_label = map_40_to_13(pr_label)

    # Definde Labels and Colors as Dictionary
    labels_and_colors = {'Bett' : 'lightblue',
                         'Bücher' : 'brown',
                         'Decke' : 'lightyellow',
                         'Stuhl' : 'orange',
                         'Fußboden' : 'magenta',
                         'Möbel' : 'blue',
                         'Objekte' : 'green',
                         'Bild' : 'red',
                         'Sofa' : 'purple',
                         'Tisch' : 'goldenrod',
                         'Fernseher' : 'lightgreen',
                         'Wand' : 'gray',
                         'Fenster' : 'lightgray',
                         'Nicht annotiert' : 'white'}

    # Create Colormap from Dictionary
    cmap = mcolors.ListedColormap(list(labels_and_colors.values()))

    # --------------------------------------------------------------------------------------------
    # Gt Label
    # --------------------------------------------------------------------------------------------
    # Convert Mask from Tensor to Image
    gt_label = gt_label.squeeze().numpy()
    # Set Unlabeled Pixels to Value 14 (For the colormap)
    gt_label[gt_label == 255] = 14

    plt.subplot(1, 3, 2)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(gt_label, cmap=cmap, vmin=0, vmax=14)

    # --------------------------------------------------------------------------------------------
    # Pr Label
    # --------------------------------------------------------------------------------------------
    # Convert Mask from Tensor to Image
    pr_label = pr_label.squeeze().numpy()
    # Set Unlabeled Pixels to Value 14 (For the colormap)
    pr_label[pr_label == 255] = 14
    plt.subplot(1, 3, 3)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(pr_label, cmap=cmap, vmin=0, vmax=14)

    # --------------------------------------------------------------------------------------------
    # Legend
    # --------------------------------------------------------------------------------------------
    legend_elements = [mpatches.Patch(facecolor=labels_and_colors[label],
                             edgecolor='black',
                             label=label) for label in labels_and_colors]
    plt.legend(handles=legend_elements,
               loc='center left',
               bbox_to_anchor=(1, 0.5))
    
    # Set Legend Title
    plt.gca().get_legend().set_title('Annotationen')

    # Set the legend font size
    plt.gca().get_legend().get_title().set_fontsize('xx-large')
    
    # Make legend bold
    plt.setp(plt.gca().get_legend().get_title(), fontweight='bold')

    # Set the font size of the labels and font type
    for label in plt.gca().get_legend().get_texts():
        label.set_fontsize('xx-large')
        label.set_fontfamily('serif')

    # Make the font of my legend look like latex
    plt.gca().get_legend().get_title().set_fontfamily('serif')

    # --------------------------------------------------------------------------------------------
    # Save Figure
    # --------------------------------------------------------------------------------------------
    directory = './figures/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(os.path.join(directory, filename))
    plt.close()


'''
Creates a 5-Tuple of image and gt label, pr label, gt depth and pr depth
'''
def visualize_img_depth(image, gt_depth, pr_depth, filename='test.png'):
    # --------------------------------------------------------------------------------------------
    # place subplots
    # --------------------------------------------------------------------------------------------
    plt.figure(figsize=(18, 5))

    plt.subplots_adjust(left=0.005,
                        bottom=0,
                        right=0.95,
                        top=1,
                        wspace=0.25,
                        hspace=0.0)

    # --------------------------------------------------------------------------------------------
    # Image
    # --------------------------------------------------------------------------------------------
    plt.subplot(1, 3, 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image.permute(1, 2, 0).numpy())

    # --------------------------------------------------------------------------------------------
    # Gt Depth Map
    # --------------------------------------------------------------------------------------------
    plt.subplot(1, 3, 2)
    plt.xticks([])
    plt.yticks([])
    cmap = plt.cm.get_cmap('Spectral')
    norm = plt.Normalize(vmin=0, vmax=10)
    plt.imshow(gt_depth.squeeze().numpy(), cmap=cmap, norm=norm)

    # --------------------------------------------------------------------------------------------
    # Pr Depth Map
    # --------------------------------------------------------------------------------------------
    plt.subplot(1, 3, 3)
    plt.xticks([])
    plt.yticks([])
    cmap = plt.cm.get_cmap('Spectral')
    norm = plt.Normalize(vmin=0, vmax=10)
    plt.imshow(pr_depth.squeeze().numpy(), cmap=cmap, norm=norm)

    # --------------------------------------------------------------------------------------------
    # Colorbar
    # --------------------------------------------------------------------------------------------
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), 
                        ax=plt.gcf().get_axes(), 
                        orientation='vertical', 
                        pad=0.02,
                        fraction=0.012)
    cbar.set_label('Depth in meters')

    # --------------------------------------------------------------------------------------------
    # Save Figure
    # --------------------------------------------------------------------------------------------
    directory = './figures'
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(os.path.join(directory, filename))
    plt.close()


'''
Creates a plot of the image, gt and pr labels and gt and pr depth maps
'''
def visualize_img_label_depth(image, gt_label, pr_label, gt_depth, pr_depth, filename='test.png'):
    # --------------------------------------------------------------------------------------------
    # Place Subplots
    # --------------------------------------------------------------------------------------------
    plt.figure(figsize=(16, 5))

    # Leave everything as it is!!!
    # If then only adjust the wspace value!!!
    plt.subplots_adjust(left=0.005,
                        bottom=0,
                        right=0.84,
                        top=1,
                        wspace=0.01,
                        hspace=0.0)
    
    # --------------------------------------------------------------------------------------------
    # Image
    # --------------------------------------------------------------------------------------------
    # Convert Image from Tensor to Image
    image = image.permute(1, 2, 0).numpy()
    plt.subplot(1, 5, 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image)

    # --------------------------------------------------------------------------------------------
    # Label preprocessing
    # --------------------------------------------------------------------------------------------
    gt_label = map_40_to_13(gt_label)
    pr_label = map_40_to_13(pr_label)

    # Definde Labels and Colors as Dictionary
    labels_and_colors = {'Bett' : 'lightblue',
                         'Bücher' : 'brown',
                         'Decke' : 'lightyellow',
                         'Stuhl' : 'orange',
                         'Fußboden' : 'magenta',
                         'Möbel' : 'blue',
                         'Objekte' : 'green',
                         'Bild' : 'red',
                         'Sofa' : 'purple',
                         'Tisch' : 'goldenrod',
                         'Fernseher' : 'lightgreen',
                         'Wand' : 'gray',
                         'Fenster' : 'lightgray',
                         'Nicht annotiert' : 'white'}

    # Create Colormap from Dictionary
    cmap = mcolors.ListedColormap(list(labels_and_colors.values()))

    # --------------------------------------------------------------------------------------------
    # Ground Truth Mask
    # --------------------------------------------------------------------------------------------
    # Convert Mask from Tensor to Image
    gt_label = gt_label.squeeze().numpy()
    # Set Unlabeled Pixels to Value 14 (For the colormap)
    gt_label[gt_label == 255] = 14

    plt.subplot(1, 5, 2)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(gt_label, cmap=cmap, vmin=0, vmax=14)

    # --------------------------------------------------------------------------------------------
    # Prediction Mask
    # --------------------------------------------------------------------------------------------
    # Convert Mask from Tensor to Image
    pr_label = pr_label.squeeze().numpy()
    # Set Unlabeled Pixels to Value 14 (For the colormap)
    pr_label[pr_label == 255] = 14
    plt.subplot(1, 5, 3)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(pr_label, cmap=cmap, vmin=0, vmax=14)

    # --------------------------------------------------------------------------------------------
    # Ground Truth Depth Map
    # --------------------------------------------------------------------------------------------
    plt.subplot(1, 5, 4)
    plt.xticks([])
    plt.yticks([])
    cmap = plt.cm.get_cmap('Spectral')
    norm = plt.Normalize(vmin=0, vmax=10)
    plt.imshow(gt_depth.squeeze().numpy(), cmap=cmap, norm=norm)

    # --------------------------------------------------------------------------------------------
    # Prediction Depth Map
    # --------------------------------------------------------------------------------------------
    plt.subplot(1, 5, 5)
    plt.xticks([])
    plt.yticks([])
    cmap = plt.cm.get_cmap('Spectral')
    norm = plt.Normalize(vmin=0, vmax=10)
    plt.imshow(pr_depth.squeeze().numpy(), cmap=cmap, norm=norm)
    
    # --------------------------------------------------------------------------------------------
    # Legend
    # --------------------------------------------------------------------------------------------
    legend_elements = [mpatches.Patch(facecolor=labels_and_colors[label],
                             edgecolor='black',
                             label=label) for label in labels_and_colors]
    plt.legend(handles=legend_elements,
               loc='center left',
               bbox_to_anchor=(1.4, 0.5))
    
    # Set Legend Title
    plt.gca().get_legend().set_title('Annotationen')

    # Set the legend font size
    plt.gca().get_legend().get_title().set_fontsize('medium')
    
    # Make legend bold
    plt.setp(plt.gca().get_legend().get_title(), fontweight='bold')

    # Set the font size of the labels and font type
    for label in plt.gca().get_legend().get_texts():
        label.set_fontsize('medium')
        label.set_fontfamily('serif')

    # Make the font of my legend look like latex
    plt.gca().get_legend().get_title().set_fontfamily('serif')

    # --------------------------------------------------------------------------------------------
    # Colorbar
    # --------------------------------------------------------------------------------------------
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), 
                        ax=plt.gcf().get_axes(), 
                        orientation='vertical', 
                        pad=0.02,
                        fraction=0.012)
    cbar.set_label('Depth in meters')

    # --------------------------------------------------------------------------------------------
    # Save Figure
    # --------------------------------------------------------------------------------------------
    directory = './figures/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(os.path.join(directory, filename))
    plt.close()