import numpy as np

import numbers


def _check_weights(weights):
    """Check to make sure weights are valid"""
    if weights not in (None, "uniform", "distance") and not callable(weights):
        raise ValueError(
            "weights not recognized: should be 'uniform', "
            "'distance', or a callable function"
        )

    return weights

def _get_weights(dist, weights):
    """Get the weights from an array of distances and a parameter ``weights``.
    Parameters
    ----------
    dist : ndarray
        The input distances.
    weights : {'uniform', 'distance' or a callable}
        The kind of weighting used.
    Returns
    -------
    weights_arr : array of the same shape as ``dist``
        If ``weights == 'uniform'``, then returns None.
    """
    if weights in (None, "uniform"):
        return None
    elif weights == "distance":
        # if user attempts to classify a point that was zero distance from one
        # or more training points, those training points are weighted as 1.0
        # and the other points as 0.0
        if dist.dtype is np.dtype(object):
            for point_dist_i, point_dist in enumerate(dist):
                # check if point_dist is iterable
                # (ex: RadiusNeighborClassifier.predict may set an element of
                # dist to 1e-6 to represent an 'outlier')
                if hasattr(point_dist, "__contains__") and 0.0 in point_dist:
                    dist[point_dist_i] = point_dist == 0.0
                else:
                    dist[point_dist_i] = 1.0 / point_dist
        else:
            with np.errstate(divide="ignore"):
                dist = 1.0 / dist
            inf_mask = np.isinf(dist)
            inf_row = np.any(inf_mask, axis=1)
            dist[inf_row] = inf_mask[inf_row]
        return dist
    elif callable(weights):
        return weights(dist)
    else:
        raise ValueError(
            "weights not recognized: should be 'uniform', "
            "'distance', or a callable function"
        )


def weighted_mode(a, w, *, axis=0):
    if axis is None:
        a = np.ravel(a)
        w = np.ravel(w)
        axis = 0
    else:
        a = np.asarray(a)
        w = np.asarray(w)

    if a.shape != w.shape:
        w = np.full(a.shape, w, dtype=w.dtype)

    scores = np.unique(np.ravel(a))  # get ALL unique values
    testshape = list(a.shape)
    testshape[axis] = 1
    oldmostfreq = np.zeros(testshape)
    oldcounts = np.zeros(testshape)
    for score in scores:
        template = np.zeros(a.shape)
        ind = a == score
        template[ind] = w[ind]
        counts = np.expand_dims(np.sum(template, axis), axis)
        mostfrequent = np.where(counts > oldcounts, score, oldmostfreq)
        oldcounts = np.maximum(counts, oldcounts)
        oldmostfreq = mostfrequent
    return mostfrequent, oldcounts

def _num_samples(x):
    """Return number of samples in array-like x."""
    message = "Expected sequence or array-like, got %s" % type(x)
    if hasattr(x, "fit") and callable(x.fit):
        # Don't get num_samples from an ensembles length!
        raise TypeError(message)

    if not hasattr(x, "__len__") and not hasattr(x, "shape"):
        if hasattr(x, "__array__"):
            x = np.asarray(x)
        else:
            raise TypeError(message)

    if hasattr(x, "shape") and x.shape is not None:
        if len(x.shape) == 0:
            raise TypeError(
                "Singleton array %r cannot be considered a valid collection." % x
            )
        # Check that shape is returning an integer or default to len
        # Dask dataframes may not return numeric shape[0] value
        if isinstance(x.shape[0], numbers.Integral):
            return x.shape[0]

    try:
        return len(x)
    except TypeError as type_error:
        raise TypeError(message) from type_error
