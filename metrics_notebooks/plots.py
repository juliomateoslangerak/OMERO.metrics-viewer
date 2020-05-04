import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from scipy.interpolate import griddata

from utils import get_tables, get_intensities

from bokeh.io import output_notebook, reset_output, show
from bokeh.layouts import gridplot
from bokeh.plotting import figure
from bokeh.models import ColorBar, LinearColorMapper, LogColorMapper, Label

reset_output()
output_notebook()  # show output in notebook


def plot_homogeneity_map(image):
    nr_channels = image.getSizeC()
    x_dim = image.getSizeX()
    y_dim = image.getSizeY()

    tables = get_tables(image, namespace_start='metrics', name_filter='properties')
    if len(tables) != 1:
        raise Exception('There are none or more than one properties tables. Verify data integrity.')
    table = tables[0]

    row_count = table.getNumberOfRows()
    col_names = [c.name for c in table.getHeaders()]
    wanted_columns = ['channel',
                      'max_intensity',
                      'mean_intensity',
                      'integrated_intensity',
                      'x_weighted_centroid',
                      'y_weighted_centroid']

    fig, axes = plt.subplots(ncols=nr_channels, nrows=3, squeeze=False, figsize=(3 * nr_channels, 9))

    for c in range(nr_channels):
        data = table.slice([col_names.index(w_col) for w_col in wanted_columns],
                           table.getWhereList(condition=f'channel=={c}', variables={}, start=0, stop=row_count, step=0))
        max_intensity = np.array([val for col in data.columns for val in col.values if col.name == 'max_intensity'])
        integrated_intensity = np.array(
            [val for col in data.columns for val in col.values if col.name == 'integrated_intensity'])
        x_positions = np.array([val for col in data.columns for val in col.values if col.name == 'x_weighted_centroid'])
        y_positions = np.array([val for col in data.columns for val in col.values if col.name == 'y_weighted_centroid'])
        grid_x, grid_y = np.mgrid[0:x_dim, 0:y_dim]
        image_intensities = get_intensities(image, c_range=c, t_range=0).max(0)

        try:
            interpolated_max_int = griddata(np.stack((x_positions, y_positions), axis=1),
                                            max_intensity, (grid_x, grid_y), method='linear')
            interpolated_intgr_int = griddata(np.stack((x_positions, y_positions), axis=1),
                                              integrated_intensity, (grid_x, grid_y), method='linear')
        except Exception as e:
            # TODO: Log a warning
            interpolated_max_int = np.zeros((256, 256))

        ax = axes.ravel()
        ax[c] = plt.subplot(3, 4, c + 1)

        ax[c].imshow(np.squeeze(image_intensities), cmap='gray')
        ax[c].set_title('MIP_' + str(c))

        ax[c + nr_channels].imshow(np.flipud(interpolated_intgr_int),
                                   extent=(0, x_dim, y_dim, 0),
                                   origin='lower',
                                   cmap=cm.hot,
                                   vmin=np.amin(integrated_intensity),
                                   vmax=np.amax(integrated_intensity))
        ax[c + nr_channels].plot(x_positions, y_positions, 'k.', ms=2)
        ax[c + nr_channels].set_title('Integrated_int_' + str(c))

        ax[c + 2 * nr_channels].imshow(np.flipud(interpolated_max_int),
                                       extent=(0, x_dim, y_dim, 0),
                                       origin='lower',
                                       cmap=cm.hot,
                                       vmin=np.amin(image_intensities),
                                       vmax=np.amax(image_intensities))
        ax[c + 2 * nr_channels].plot(x_positions, y_positions, 'k.', ms=2)
        ax[c + 2 * nr_channels].set_title('Max_int_' + str(c))

    plt.show()


def plot_distances_map(image):
    nr_channels = image.getSizeC()
    x_dim = image.getSizeX()
    y_dim = image.getSizeY()

    tables = get_tables(image, namespace_start='metrics', name_filter='distances')
    if len(tables) != 1:
        raise Exception('There are none or more than one distances tables. Verify data integrity.')
    table = tables[0]
    row_count = table.getNumberOfRows()
    col_names = [c.name for c in table.getHeaders()]

    # We need the positions too
    pos_tables = get_tables(image, namespace_start='metrics', name_filter='properties')
    if len(tables) != 1:
        raise Exception('There are none or more than one positions tables. Verify data integrity.')
    pos_table = pos_tables[0]
    pos_row_count = pos_table.getNumberOfRows()
    pos_col_names = [c.name for c in pos_table.getHeaders()]

    fig, axes = plt.subplots(ncols=nr_channels - 1, nrows=nr_channels, squeeze=False,
                             figsize=((nr_channels - 1) * 3, nr_channels * 3))

    ax_index = 0
    for ch_A in range(nr_channels):
        pos_data = pos_table.slice([pos_col_names.index(w_col) for w_col in ['channel',
                                                                             'mask_labels',
                                                                             'x_weighted_centroid',
                                                                             'y_weighted_centroid']],
                                   pos_table.getWhereList(condition=f'channel=={ch_A}', variables={}, start=0,
                                                          stop=pos_row_count, step=0))

        mask_labels = np.array(
            [val for col in pos_data.columns for val in col.values if col.name == 'mask_labels'])
        x_positions = np.array(
            [val for col in pos_data.columns for val in col.values if col.name == 'x_weighted_centroid'])
        y_positions = np.array(
            [val for col in pos_data.columns for val in col.values if col.name == 'y_weighted_centroid'])
        positions_map = np.stack((x_positions, y_positions), axis=1)

        for ch_B in [i for i in range(nr_channels) if i != ch_A]:
            data = table.slice(list(range(len(col_names))),
                               table.getWhereList(condition=f'(channel_A=={ch_A})&(channel_B=={ch_B})', variables={},
                                                  start=0, stop=row_count, step=0))
            labels_map = np.array([val for col in data.columns for val in col.values if col.name == 'ch_A_roi_labels'])
            labels_map += 1  # Mask labels are augmented by one as 0 is background
            distances_map_3d = np.array(
                [val for col in data.columns for val in col.values if col.name == 'distance_3d'])
            distances_map_x = np.array([val for col in data.columns for val in col.values if col.name == 'distance_x'])
            distances_map_y = np.array([val for col in data.columns for val in col.values if col.name == 'distance_y'])
            distances_map_z = np.array([val for col in data.columns for val in col.values if col.name == 'distance_z'])

            filtered_positions = positions_map[
                                 np.intersect1d(mask_labels, labels_map, assume_unique=True, return_indices=True)[1], :]

            grid_x, grid_y = np.mgrid[0:x_dim:1, 0:y_dim:1]
            interpolated = griddata(filtered_positions, distances_map_3d, (grid_x, grid_y), method='cubic')

            ax = axes.ravel()
            ax[ax_index].imshow(np.flipud(interpolated),
                                extent=(0, x_dim, y_dim, 0),
                                origin='lower',
                                cmap=cm.hot,
                                vmin=np.amin(distances_map_3d),
                                vmax=np.amax(distances_map_3d)
                                )
            ax[ax_index].set_title(f'Distance Ch{ch_A}-Ch{ch_B}')

            ax_index += 1

    plt.show()


def plot_distances_map_bokeh(image):
    nr_channels = image.getSizeC()
    x_dim = image.getSizeX()
    y_dim = image.getSizeY()

    tables = get_tables(image, namespace_start='metrics', name_filter='distances')
    if len(tables) != 1:
        raise Exception('There are none or more than one distances tables. Verify data integrity.')
    table = tables[0]
    row_count = table.getNumberOfRows()
    col_names = [c.name for c in table.getHeaders()]

    # We need the positions too
    pos_tables = get_tables(image, namespace_start='metrics', name_filter='properties')
    if len(tables) != 1:
        raise Exception('There are none or more than one positions tables. Verify data integrity.')
    pos_table = pos_tables[0]
    pos_row_count = pos_table.getNumberOfRows()
    pos_col_names = [c.name for c in pos_table.getHeaders()]

    # Prepare the plot
    plots = [[] for x in range(nr_channels)]
    # output_file("distances_map.html", title=f"Distances map for {image.getName()}\nAcquisition date: {image.getAcquisitionDate()}")
    color_mapper = LinearColorMapper(palette="Inferno256", low=0, high=1)
    color_bar = ColorBar(color_mapper=color_mapper, label_standoff=10, location=(0, 0))

    for ch_A in range(nr_channels):
        pos_data = pos_table.slice([pos_col_names.index(w_col) for w_col in ['channel',
                                                                             'mask_labels',
                                                                             'x_weighted_centroid',
                                                                             'y_weighted_centroid']],
                                   pos_table.getWhereList(condition=f'channel=={ch_A}', variables={}, start=0,
                                                          stop=pos_row_count, step=0))

        mask_labels = np.array(
            [val for col in pos_data.columns for val in col.values if col.name == 'mask_labels'])
        x_positions = np.array(
            [val for col in pos_data.columns for val in col.values if col.name == 'x_weighted_centroid'])
        y_positions = np.array(
            [val for col in pos_data.columns for val in col.values if col.name == 'y_weighted_centroid'])
        positions_map = np.stack((x_positions, y_positions), axis=1)

        for ch_B in [i for i in range(nr_channels) if i != ch_A]:
            data = table.slice(list(range(len(col_names))),
                               table.getWhereList(condition=f'(channel_A=={ch_A})&(channel_B=={ch_B})', variables={},
                                                  start=0, stop=row_count, step=0))
            labels_map = np.array([val for col in data.columns for val in col.values if col.name == 'ch_A_roi_labels'])
            labels_map += 1  # Mask labels are augmented by one as 0 is background
            distances_map_3d = np.array(
                [val for col in data.columns for val in col.values if col.name == 'distance_3d'])
            distances_map_x = np.array([val for col in data.columns for val in col.values if col.name == 'distance_x'])
            distances_map_y = np.array([val for col in data.columns for val in col.values if col.name == 'distance_y'])
            distances_map_z = np.array([val for col in data.columns for val in col.values if col.name == 'distance_z'])

            filtered_positions = positions_map[
                                 np.intersect1d(mask_labels, labels_map, assume_unique=True, return_indices=True)[1], :]

            grid_x, grid_y = np.mgrid[0:x_dim:1, 0:y_dim:1]
            interpolated = griddata(filtered_positions, distances_map_3d, (grid_x, grid_y), method='cubic')

            p = figure()
            p.title.text = f'Distance Ch{ch_A}-Ch{ch_B}'
            p.title.align = 'center'
            p.title.text_font_size = '18px'
            p.image(image=[interpolated],
                    x=0, y=0, dw=x_dim, dh=y_dim,
                    color_mapper=color_mapper)
            p.add_layout(color_bar, 'right')

            plots[ch_A].append(p)

    grid = gridplot(plots, plot_width=200 * (nr_channels - 1),
                    sizing_mode='scale_both')  # , plot_height=100 * nr_channels)

    show(grid)


def plot_psfs(image, bead_nr):
    def plot_mip(image, title):
        fig = figure()
        fig.title.text = title
        fig.title.align = 'center'
        fig.image(image=[image], x=0, y=0, dw=image.shape[0], dh=image.shape[1], palette="Inferno256")

        return fig

    def plot_prof(profiles, title, fwhm):
        fig = figure()
        fig.title.text = title
        fig.title.align = 'center'
        fig.line(range(profiles[0].shape[0]), profiles[0], line_width=2, line_color="navy", )
        fig.line(range(profiles[1].shape[0]), profiles[1], line_width=2, line_color='red', line_dash='dashed')
        fig.add_layout(Label(x=0, y=profiles[0].max() / 2, text=f'{fwhm:.3f}',
                             background_fill_color='white', background_fill_alpha=.6))

        return fig

    pixel_size = (image.getPixelSizeZ(),
                  image.getPixelSizeY(),
                  image.getPixelSizeX())

    properties_tables = get_tables(image, namespace_start='metrics', name_filter='properties')
    if len(properties_tables) != 1:
        raise Exception('There are none or more than one distances tables. Verify data integrity.')
    properties_table = properties_tables[0]
    properties_row_count = properties_table.getNumberOfRows()
    properties_col_names = [c.name for c in properties_table.getHeaders()]
    properties_data = properties_table.readCoordinates(list(range(properties_row_count)))

    x_profiles_table = get_tables(image, namespace_start='metrics', name_filter='X_profiles')[0]
    x_profiles_data = x_profiles_table.readCoordinates(list(range(x_profiles_table.getNumberOfRows())))

    y_profiles_table = get_tables(image, namespace_start='metrics', name_filter='Y_profiles')[0]
    y_profiles_data = y_profiles_table.readCoordinates(list(range(y_profiles_table.getNumberOfRows())))

    z_profiles_table = get_tables(image, namespace_start='metrics', name_filter='Z_profiles')[0]
    z_profiles_data = z_profiles_table.readCoordinates(list(range(z_profiles_table.getNumberOfRows())))

    psf_images = list(image._conn.getObjects('Image',
                                             ids=[val for col in properties_data.columns for val in col.values if
                                                  col.name == 'bead_image']))
    x_fwhms = [val for col in properties_data.columns for val in col.values if col.name == 'x_fwhm']
    y_fwhms = [val for col in properties_data.columns for val in col.values if col.name == 'y_fwhm']
    z_fwhms = [val for col in properties_data.columns for val in col.values if col.name == 'z_fwhm']
    fwhm_units = [val for col in properties_data.columns for val in col.values if col.name == 'fwhm_units'][0]

    # Prepare the plot
    plots = [[] for x in range(properties_row_count * 3)]
    # output_file("psfs.html", title=f"PSFs for image {image.getName()}\nAcquisition date: {image.getAcquisitionDate()}")
    # color_bar = ColorBar(color_mapper=color_mapper, label_standoff=10, location=(0, 0))

    if bead_nr is None or bead_nr == 'all':
        start = 0
        end = len(psf_images)
    elif bead_nr == 0:
        return
    elif bead_nr > 0:
        start = bead_nr - 1
        end = bead_nr

    for i, (psf_image, x_fwhm, y_fwhm, z_fwhm) in enumerate(zip(psf_images[start:end], x_fwhms[start:end], y_fwhms[start:end], z_fwhms[start:end])):
        psf_intensities = np.squeeze(get_intensities(psf_image))
        x_dim = psf_image.getSizeX()
        y_dim = psf_image.getSizeY()
        z_dim = psf_image.getSizeZ()
        x_mip = psf_intensities.max(axis=2)
        y_mip = psf_intensities.max(axis=1)
        z_mip = psf_intensities.max(axis=0)
        x_raw_profile = np.array(
            [val for col in x_profiles_data.columns for val in col.values if col.name == f'raw_x_profile_bead-{i:02d}'])
        x_fitted_profile = np.array([val for col in x_profiles_data.columns for val in col.values if
                                     col.name == f'fitted_x_profile_bead-{i:02d}'])
        y_raw_profile = np.array(
            [val for col in y_profiles_data.columns for val in col.values if col.name == f'raw_y_profile_bead-{i:02d}'])
        y_fitted_profile = np.array([val for col in y_profiles_data.columns for val in col.values if
                                     col.name == f'fitted_y_profile_bead-{i:02d}'])
        z_raw_profile = np.array(
            [val for col in z_profiles_data.columns for val in col.values if col.name == f'raw_z_profile_bead-{i:02d}'])
        z_fitted_profile = np.array([val for col in z_profiles_data.columns for val in col.values if
                                     col.name == f'fitted_z_profile_bead-{i:02d}'])

        color_mapper = LogColorMapper(palette="Inferno256", low=0, high=psf_intensities.max())

        x_mip_fig = plot_mip(x_mip, f'X MIP bead {i+1}')
        y_mip_fig = plot_mip(y_mip, f'Y MIP bead {i+1}')
        z_mip_fig = plot_mip(z_mip, f'Z MIP bead {i+1}')

        x_prof_fig = plot_prof((x_raw_profile, x_fitted_profile), f'X profile bead {i}', x_fwhm)
        y_prof_fig = plot_prof((y_raw_profile, y_fitted_profile), f'Y profile bead {i}', y_fwhm)
        z_prof_fig = plot_prof((z_raw_profile, z_fitted_profile), f'Z profile bead {i}', z_fwhm)

        plots[i*3].extend([x_mip_fig, x_prof_fig])
        plots[(i*3)+1].extend([y_mip_fig, y_prof_fig])
        plots[(i*3)+2].extend([z_mip_fig, z_prof_fig])

    grid = gridplot(plots, plot_width=600, sizing_mode='scale_both')  # , plot_height=100 * nr_channels)

    show(grid)

