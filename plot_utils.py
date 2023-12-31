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
Returns a dictionary of labels and colors for the segmentation
'''
def get_labels_and_colors():
    labels_and_colors = {'Bett' : (0,0,1),
                         'Bücher' : (0.9137,0.3490,0.1882),
                         'Decke' : (0, 0.8549, 0),
                         'Stuhl' : (0.5843,0,0.9412),
                         'Fußboden' : (0.8706,0.9451,0.0941),
                         'Möbel' : (1.0000,0.8078,0.8078),
                         'Objekte' : (0,0.8784,0.8980),
                         'Bild' : (0.4157,0.5333,0.8000),
                         'Sofa' : (0.4588,0.1137,0.1608),
                         'Tisch' : (0.9412,0.1373,0.9216),
                         'Fernseher' : (0,0.6549,0.6118),
                         'Wand' : (0.9765,0.5451,0),
                         'Fenster' : (0.8824,0.8980,0.7608),
                         'Nicht annotiert' : (1,1,1)}
    return labels_and_colors

'''
Creates a 2x3 plot of gts and 2 predictions
'''
def visualize_gts_prs(image, gt_label, gt_depth, pr_label_mod1, pr_depth_mod2, pr_label_mod3, pr_depth_mod3, filename='test.png'):
    # --------------------------------------------------------------------------------------------
    # place subplots
    # --------------------------------------------------------------------------------------------
    plt.figure(figsize=(16, 4.75), frameon=False)
    plt.subplots_adjust(left=0,
                        bottom=0,
                        right=1,
                        top=1,
                        wspace=0.03,
                        hspace=0.03)
    
    # --------------------------------------------------------------------------------------------
    # Image
    # --------------------------------------------------------------------------------------------
    # Convert Image from Tensor to Image
    image = image.permute(1, 2, 0).numpy()

    # Definieren Sie die Gesamtgröße des Grids
    grid_size = (2, 5)

    # Erstellen Sie ein Subplot für das Bild, das sich über beide Zeilen erstreckt, in der ersten Spalte
    ax1 = plt.subplot2grid(grid_size, (0, 0), rowspan=2, colspan=2)
    plt.xticks([])
    plt.yticks([])
    ax1.imshow(image)
    ax1.axis('off')

    # --------------------------------------------------------------------------------------------
    # Label Preprocessing
    # --------------------------------------------------------------------------------------------
    gt_label = map_40_to_13(gt_label)
    pr_label_mod1 = map_40_to_13(pr_label_mod1)
    pr_label_mod3 = map_40_to_13(pr_label_mod3)

    # Definde Labels and Colors as Dictionary --> Official Colors: https://github.com/ankurhanda/nyuv2-meta-data/issues/4
    labels_and_colors = get_labels_and_colors()

    # Create Colormap from Dictionary
    cmap = mcolors.ListedColormap(list(labels_and_colors.values()))

    # --------------------------------------------------------------------------------------------
    # Gt Label
    # --------------------------------------------------------------------------------------------
    # Convert Mask from Tensor to Image
    gt_label = gt_label.squeeze().numpy()
    # Set Unlabeled Pixels Value 14 (For the colormap)
    gt_label[gt_label == 255] = 14

    plt.subplot(2, 5, 3)
    plt.xticks([])
    plt.yticks([])

    # Plot the original image in the background and the transparent mask in the foreground
    #plt.imshow(image)
    plt.imshow(gt_label, cmap=cmap, vmin=0, vmax=14)#, alpha=0.5)
    plt.axis('off')

    # --------------------------------------------------------------------------------------------
    # Pr Label  Model 1
    # --------------------------------------------------------------------------------------------
    # Convert Mask from Tensor to Image
    pr_label_mod1 = pr_label_mod1.squeeze().numpy()
    # Set Unlabeled Pixels to Value 14 (For the colormap)
    pr_label_mod1[pr_label_mod1 == 255] = 14
    plt.subplot(2, 5, 4)
    plt.xticks([])
    plt.yticks([])
    #plt.imshow(image)
    plt.imshow(pr_label_mod1, cmap=cmap, vmin=0, vmax=14)#, alpha=0.5)
    plt.axis('off')

    # --------------------------------------------------------------------------------------------
    # Pr Label  Model 3
    # --------------------------------------------------------------------------------------------
    # Convert Mask from Tensor to Image
    pr_label_mod3 = pr_label_mod3.squeeze().numpy()
    # Set Unlabeled Pixels to Value 14 (For the colormap)
    pr_label_mod3[pr_label_mod3 == 255] = 14
    plt.subplot(2, 5, 5)
    plt.xticks([])
    plt.yticks([])
    #plt.imshow(image)
    plt.imshow(pr_label_mod3, cmap=cmap, vmin=0, vmax=14)#, alpha=0.5)
    plt.axis('off')

    # --------------------------------------------------------------------------------------------
    # Gt Depth Map
    # --------------------------------------------------------------------------------------------
    plt.subplot(2, 5, 8)
    plt.xticks([])
    plt.yticks([])
    cmap = plt.cm.get_cmap('plasma_r')
    norm = plt.Normalize(vmin=0, vmax=10)
    plt.imshow(gt_depth.squeeze().numpy(), cmap=cmap, norm=norm)
    plt.axis('off')

    # --------------------------------------------------------------------------------------------
    # Pr Depth Map Model 2
    # --------------------------------------------------------------------------------------------
    plt.subplot(2, 5, 9)
    plt.xticks([])
    plt.yticks([])
    cmap = plt.cm.get_cmap('plasma_r')
    norm = plt.Normalize(vmin=0, vmax=10)
    plt.imshow(pr_depth_mod2.squeeze().numpy(), cmap=cmap, norm=norm)
    plt.axis('off')

    # --------------------------------------------------------------------------------------------
    # Pr Depth Map Model 3
    # --------------------------------------------------------------------------------------------
    plt.subplot(2, 5, 10)
    plt.xticks([])
    plt.yticks([])
    cmap = plt.cm.get_cmap('plasma_r')
    norm = plt.Normalize(vmin=0, vmax=10)
    plt.imshow(pr_depth_mod3.squeeze().numpy(), cmap=cmap, norm=norm)
    plt.axis('off')

    # --------------------------------------------------------------------------------------------
    # Save Figure
    # --------------------------------------------------------------------------------------------
    directory = './figures/discussion/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(os.path.join(directory, filename))
    plt.close()


'''
Creates a triple of image, GT mask and segmented mask and saves it to directory figures
'''
def visualize_img_label(image, gt_label, pr_label, filename='test.png'):
    # --------------------------------------------------------------------------------------------
    # place subplots
    # --------------------------------------------------------------------------------------------
    plt.figure(figsize=(16, 3.88), frameon=False)

    # Leave everything as it is!!!
    # If then only adjust the wspace value!!!
    plt.subplots_adjust(left=0,
                        bottom=0,
                        right=1,
                        top=1,
                        wspace=0.05,
                        hspace=0.0)
    
    # --------------------------------------------------------------------------------------------
    # Image
    # --------------------------------------------------------------------------------------------
    # Convert Image from Tensor to Image
    image = image.permute(1, 2, 0).numpy()
    # Make sure that matplotlib doesnt give the image a black border
    # https://stackoverflow.com/questions/8218608/scipy-savefig-without-frames-axes-only-content
    plt.subplot(1, 3, 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image)
    plt.axis('off')

    # --------------------------------------------------------------------------------------------
    # Label Preprocessing
    # --------------------------------------------------------------------------------------------
    gt_label = map_40_to_13(gt_label)
    pr_label = map_40_to_13(pr_label)

    # Definde Labels and Colors as Dictionary --> Official Colors: https://github.com/ankurhanda/nyuv2-meta-data/issues/4
    labels_and_colors = get_labels_and_colors()

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

    # Plot the original image in the background and the transparent mask in the foreground
    #plt.imshow(image)
    plt.imshow(gt_label, cmap=cmap, vmin=0, vmax=14)#, alpha=0.5)
    plt.axis('off')

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
    #plt.imshow(image)
    plt.imshow(pr_label, cmap=cmap, vmin=0, vmax=14)#, alpha=0.5)
    plt.axis('off')
    '''
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
    '''
    # --------------------------------------------------------------------------------------------
    # Save Figure
    # --------------------------------------------------------------------------------------------
    directory = './figures/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(os.path.join(directory, filename))
    plt.close()


'''
Creates a Triple of image, gt depth and pr depth
'''
def visualize_img_depth(image, gt_depth, pr_depth, filename='test.png'):
    # --------------------------------------------------------------------------------------------
    # place subplots
    # --------------------------------------------------------------------------------------------
    plt.figure(figsize=(16, 3.88), frameon=False)

    plt.subplots_adjust(left=0,
                        bottom=0,
                        right=1,
                        top=1,
                        wspace=0.05,
                        hspace=0.0)

    # --------------------------------------------------------------------------------------------
    # Image
    # --------------------------------------------------------------------------------------------
    image = image.permute(1, 2, 0).numpy()
    plt.subplot(1, 3, 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image)
    plt.axis('off')

    # --------------------------------------------------------------------------------------------
    # Gt Depth Map
    # --------------------------------------------------------------------------------------------
    plt.subplot(1, 3, 2)
    plt.xticks([])
    plt.yticks([])
    cmap = plt.cm.get_cmap('plasma_r')
    norm = plt.Normalize(vmin=0, vmax=10)
    plt.imshow(gt_depth.squeeze().numpy(), cmap=cmap, norm=norm)
    plt.axis('off')

    # --------------------------------------------------------------------------------------------
    # Pr Depth Map
    # --------------------------------------------------------------------------------------------
    plt.subplot(1, 3, 3)
    plt.xticks([])
    plt.yticks([])
    cmap = plt.cm.get_cmap('plasma_r')
    norm = plt.Normalize(vmin=0, vmax=10)
    plt.imshow(pr_depth.squeeze().numpy(), cmap=cmap, norm=norm)
    plt.axis('off')
    '''
    # --------------------------------------------------------------------------------------------
    # Colorbar
    # --------------------------------------------------------------------------------------------
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), 
                        ax=plt.gcf().get_axes(), 
                        orientation='vertical', 
                        pad=0.02,
                        fraction=0.012)
    cbar.set_label('Depth in meters')
    '''
    # --------------------------------------------------------------------------------------------
    # Save Figure
    # --------------------------------------------------------------------------------------------
    directory = './figures'
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(os.path.join(directory, filename))
    plt.close()


'''
Creates a plot of the image, gt labels and gt depth maps
'''
def visualize_img_gts(image, gt_label, gt_depth, filename='test.png'):
    # --------------------------------------------------------------------------------------------
    # place subplots
    # --------------------------------------------------------------------------------------------
    plt.figure(figsize=(16, 4.5), frameon=False)

    # Leave everything as it is!!!
    # If then only adjust the wspace value!!!
    plt.subplots_adjust(left=0,
                        bottom=0,
                        right=0.79,
                        top=1,
                        wspace=0.05,
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
    plt.axis('off')

    # --------------------------------------------------------------------------------------------
    # Label Preprocessing
    # --------------------------------------------------------------------------------------------
    gt_label = map_40_to_13(gt_label)

    # Definde Labels and Colors as Dictionary
    labels_and_colors = get_labels_and_colors()

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
    #plt.imshow(image)
    plt.imshow(gt_label, cmap=cmap, vmin=0, vmax=14)#, alpha=0.5)
    plt.axis('off')

    # --------------------------------------------------------------------------------------------
    # Gt Depth
    # --------------------------------------------------------------------------------------------
    plt.subplot(1, 3, 3)
    plt.xticks([])
    plt.yticks([])
    cmap = plt.cm.get_cmap('plasma_r')
    norm = plt.Normalize(vmin=0, vmax=10)
    plt.imshow(gt_depth.squeeze().numpy(), cmap=cmap, norm=norm)
    plt.axis('off')

    # --------------------------------------------------------------------------------------------
    # Legend
    # --------------------------------------------------------------------------------------------
    legend_elements = [mpatches.Patch(facecolor=labels_and_colors[label],
                             edgecolor='black',
                             label=label) for label in labels_and_colors]
    plt.legend(handles=legend_elements,
               loc='center left',
               bbox_to_anchor=(1.35, 0.5))
    
    # Set Legend Title
    #plt.gca().get_legend().set_title('Annotationen')

    # Set the legend font size
    #plt.gca().get_legend().get_title().set_fontsize('medium')
    
    # Make legend bold
    #plt.setp(plt.gca().get_legend().get_title(), fontweight='bold')

    # Set the font size of the labels and font type
    for label in plt.gca().get_legend().get_texts():
        label.set_fontsize('xx-large')
        #label.set_fontfamily('serif')

    # Make the font of my legend look like latex
    #plt.gca().get_legend().get_title().set_fontfamily('serif')

    # --------------------------------------------------------------------------------------------
    # Colorbar
    # --------------------------------------------------------------------------------------------
    # Colorbar
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), 
                    ax=plt.gcf().get_axes(), 
                    orientation='vertical', 
                    pad=0.02,
                    fraction=0.0165)
    cbar.set_label('Tiefe (m)', size='xx-large')
    cbar.ax.tick_params(labelsize='xx-large')

    # --------------------------------------------------------------------------------------------
    # Save Figure
    # --------------------------------------------------------------------------------------------
    directory = './figures/'
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
    plt.figure(figsize=(16, 2.325), frameon=False)

    # Leave everything as it is!!!
    # If then only adjust the wspace value!!!
    plt.subplots_adjust(left=0,
                        bottom=0,
                        right=1,
                        top=1,
                        wspace=0.05,
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
    plt.axis('off')

    # --------------------------------------------------------------------------------------------
    # Label preprocessing
    # --------------------------------------------------------------------------------------------
    gt_label = map_40_to_13(gt_label)
    pr_label = map_40_to_13(pr_label)

    # Definde Labels and Colors as Dictionary
    labels_and_colors = get_labels_and_colors()

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
    plt.imshow(image)
    plt.imshow(gt_label, cmap=cmap, vmin=0, vmax=14)#, alpha=0.5)
    plt.axis('off')

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
    plt.imshow(image)
    plt.imshow(pr_label, cmap=cmap, vmin=0, vmax=14)#, alpha=0.5)
    plt.axis('off')

    # --------------------------------------------------------------------------------------------
    # Ground Truth Depth Map
    # --------------------------------------------------------------------------------------------
    plt.subplot(1, 5, 4)
    plt.xticks([])
    plt.yticks([])
    cmap = plt.cm.get_cmap('plasma_r')
    norm = plt.Normalize(vmin=0, vmax=10)
    plt.imshow(gt_depth.squeeze().numpy(), cmap=cmap, norm=norm)
    plt.axis('off')

    # --------------------------------------------------------------------------------------------
    # Prediction Depth Map
    # --------------------------------------------------------------------------------------------
    plt.subplot(1, 5, 5)
    plt.xticks([])
    plt.yticks([])
    cmap = plt.cm.get_cmap('plasma_r')
    norm = plt.Normalize(vmin=0, vmax=10)
    plt.imshow(pr_depth.squeeze().numpy(), cmap=cmap, norm=norm)
    plt.axis('off')
    '''
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
    '''
    # --------------------------------------------------------------------------------------------
    # Save Figure
    # --------------------------------------------------------------------------------------------
    directory = './figures/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(os.path.join(directory, filename))
    plt.close()