import matplotlib.pyplot as plt
#from mask_data import mask_list

def show_masks(masks):
    '''
    Masks is a list of arrays containing images of masks. Last two elements are
    the epoch at which they were pulled and the image id.
    '''
    output, global_mask_otsu, global_mask_hysteresis, global_mask_fill, epoch, image_id = masks

    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax[0, 0].imshow(output, cmap='gray')
    ax[0, 0].set_title('Net output')

    ax[0, 1].imshow(global_mask_otsu, cmap='magma')
    ax[0, 1].set_title('Otsu mask')

    ax[1, 0].imshow(global_mask_hysteresis, cmap='magma')
    ax[1, 0].set_title('Hysteresis mask')

    ax[1, 1].imshow(global_mask_fill, cmap='magma')
    ax[1, 1].set_title('Hysteresis with binary fill'

    fig.suptitle('Epoch {}, image id = {}'.format(epoch, image_id[-5:]))
    plt.show()

if __name__ == '__main__':
    for masks in mask_list:
        show_masks(masks)
