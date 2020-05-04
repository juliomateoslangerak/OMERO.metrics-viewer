import omero.gateway as gw
from itertools import product
import numpy as np


def get_image_shape(image):
    try:
        image_shape = (image.getSizeZ(),
                       image.getSizeC(),
                       image.getSizeT(),
                       image.getSizeY(),
                       image.getSizeX())
    except Exception as e:
        raise e

    return image_shape


def get_intensities(image, z_range=None, c_range=None, t_range=None, x_range=None, y_range=None):
    """Returns a numpy array containing the intensity values of the image
    Returns an array with dimensions arranged as zctxy
    """
    image_shape = get_image_shape(image)

    # Decide if we are going to call getPlanes or getTiles
    if not x_range and not y_range:
        whole_planes = True
    else:
        whole_planes = False

    ranges = list(range(5))
    for dim, r in enumerate([z_range, c_range, t_range, y_range, x_range]):
        # Verify that requested ranges are within the available data
        if r is None:  # Range is not specified
            ranges[dim] = range(image_shape[dim])
        else:  # Range is specified
            if type(r) is int:
                ranges[dim] = range(r, r + 1)
            elif type(r) is not tuple:
                raise TypeError('Range is not provided as a tuple.')
            else:  # range is a tuple
                if len(r) == 1:
                    ranges[dim] = range(r[0])
                elif len(r) == 2:
                    ranges[dim] = range(r[0], r[1])
                elif len(r) == 3:
                    ranges[dim] = range(r[0], r[1], r[2])
                else:
                    raise IndexError('Range values must contain 1 to three values')
            if not 1 <= ranges[dim].stop <= image_shape[dim]:
                raise IndexError('Specified range is outside of the image dimensions')

    output_shape = (len(ranges[0]), len(ranges[1]), len(ranges[2]), len(ranges[3]), len(ranges[4]))
    nr_planes = output_shape[0] * output_shape[1] * output_shape[2]
    zct_list = list(product(ranges[0], ranges[1], ranges[2]))

    pixels = image.getPrimaryPixels()
    pixels_type = pixels.getPixelsType()
    if pixels_type.value == 'float':
        data_type = pixels_type.value + str(pixels_type.bitSize)  # TODO: Verify this is working for all data types
    else:
        data_type = pixels_type.value

    intensities = np.zeros((nr_planes,
                            output_shape[3],
                            output_shape[4]),
                           dtype=data_type)
    if whole_planes:
        np.stack(list(pixels.getPlanes(zctList=zct_list)), out=intensities)
    else:
        tile_region = (ranges[3].start, ranges[4].start, len(ranges[3]), len(ranges[4]))
        zct_tile_list = [(z, c, t, tile_region) for z, c, t in zct_list]
        np.stack(list(pixels.getTiles(zctTileList=zct_tile_list)), out=intensities)

    intensities = np.reshape(intensities, newshape=output_shape)

    return intensities


def get_tables(omero_object, namespace_start='', name_filter=''):
    tables_list = list()
    resources = omero_object._conn.getSharedResources()
    for ann in omero_object.listAnnotations():
        if isinstance(ann, gw.FileAnnotationWrapper) and \
                ann.getNs().startswith(namespace_start) and \
                name_filter in ann.getFileName():
            table_file = omero_object._conn.getObject("OriginalFile", attributes={'name': ann.getFileName()})
            table = resources.openTable(table_file._obj)
            tables_list.append(table)

    return tables_list

