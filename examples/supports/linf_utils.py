from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib
import matplotlib as mpl

import warnings
warnings.filterwarnings("ignore")


# https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current Axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(range(data.shape[1]), labels=col_labels,
                  rotation=-30, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def get_heatmap_array(
    df: pd.DataFrame,
    race_values: list = ['Green','Blue','Purple'],
    age_values: list = ['0-18','18-30','30-45','45-60','60+'],
    protected_list: list = ["Race", "Age"],
    ) -> np.ndarray:
    '''Retrieve a matrix with the subgroups' relative frequencies
    '''
    tot_counts = sum(df.value_counts().values)
    dummy_lst1 = []
    for i in race_values:
        dummy_lst2 = []
        for j in age_values:
            dummy_v = df[(df[protected_list[0]] == i) &
                         (df[protected_list[1]] == j)]
            dummy_lst2.append(sum(dummy_v.value_counts().values))
        dummy_lst1.append(dummy_lst2)

    return np.asarray(dummy_lst1) / tot_counts


def load_heatmap(
    csv_path: str,
    target_col: str,
    protected_list: list = ["Race", "Age"],
    feature_involved: str = 'Gender',
    subgroup_to_check: str = 'F',
    ):
    '''Plot example heatmaps
    '''
    csv_path = Path(csv_path)

    df = pd.read_csv(csv_path)

    # Filter by positive target
    df = df[df[target_col] == 1]
    # (possibly) discriminated group
    discr = df[df[feature_involved] == subgroup_to_check]


    fig, (ax, ax2) = plt.subplots(1,2, figsize=(20,5))

    im, cbar = heatmap(get_heatmap_array(df), df[protected_list[0]].unique(),
                       df[protected_list[1]].unique(), ax=ax,
                       cmap="YlGn", cbarlabel="rel. freq general population")
    texts = annotate_heatmap(im, valfmt="{x:.2f}")

    im, cbar = heatmap(get_heatmap_array(discr), df[protected_list[0]].unique(),
                       df[protected_list[1]].unique(), ax=ax2,
                       cmap="YlGn", cbarlabel="rel. freq. Female instances")
    texts = annotate_heatmap(im, valfmt="{x:.2f}")

    fig.tight_layout()
    plt.show()
