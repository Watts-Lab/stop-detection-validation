from collections import defaultdict
import pandas as pd
import numpy as np
import numpy.random as npr

def extract_middle(data):
    """
    TODO
    
    Parameters
    ----------
    df : pandas.DataFrame
        User pings with 'x' (EPSG:3857), 'y' (EPSG:3857) columns, indexed by 'unix_timestamp'

    Returns
    -------
    tuple (i,j)
        First and last indices of the "middle" of the cluster
    """    
    current = data.iloc[0]['cluster']     
    x = (data.cluster != current).values  
    if len(np.where(x)[0]) == 0:        # There is no inbetween
        return(len(data), len(data))
    else:
        i = np.where(x)[0][0]           # First index where the cluster is not the value of the first entry's cluster
    if len(np.where(~x[i:])[0]) == 0:   # There is no current again (i.e., the first cluster does not reappear, so the middle is actually the tail)
        return(i, len(data))
    else:                               # Current reappears
        j = i + np.where(~x[i:])[0][0]
    return (i, j)

def find_neighbors(data, time_thresh, dist_thresh):
    """
    Identifies neighboring pings for each user ping within specified time and distance thresholds.

    Parameters
    ----------
    df : pandas.DataFrame
        User pings with 'x' (EPSG:3857), 'y' (EPSG:3857) columns, indexed by 'unix_timestamp'
    time_thresh : int
        Time threshold in minutes.
    dist_thresh : float
        Distance threshold in meters.

    Returns
    -------
    dict
        Neighbors indexed by unix timestamp, with values as sets of neighboring unix timestamps.
    """    
    
    unix_timestamps, x, y = data.index.values, data['x'].values, data['y'].values

    # Time threshold calculation using broadcasting
    within_threshold = np.triu(np.abs(unix_timestamps[:, np.newaxis] - unix_timestamps) <= (time_thresh * 60), k=1)
    t_pairs = np.where(within_threshold)

    # Distance calculation
    distances_sq = (x[t_pairs[0]] - x[t_pairs[1]])**2 + (y[t_pairs[0]] - y[t_pairs[1]])**2
    neighbor_pairs = distances_sq < dist_thresh**2

    # Building the neighbor dictionary
    neighbor_dict = defaultdict(set)
    for i, j in zip(t_pairs[0][neighbor_pairs], t_pairs[1][neighbor_pairs]):
        neighbor_dict[unix_timestamps[i].item()].add(unix_timestamps[j].item())
        neighbor_dict[unix_timestamps[j].item()].add(unix_timestamps[i].item())

    return neighbor_dict

def dbscan(data, time_thresh, dist_thresh, min_pts, neighbor_dict=None):
    """
    Implements DBSCAN.

    Parameters
    ----------
    data : pandas.DataFrame
        User pings with 'x' (EPSG:3857), 'y' (EPSG:3857) columns, indexed by 'unix_timestamp'
    time_thresh : int
        Time threshold in minutes.
    dist_thresh : float
        Distance threshold in meters.
    min_pts: int
        A cluster must have at least (min_pts+1) points to be considered a cluster.

    Returns
    -------
    pandas.DataFrame
        Contains two columns 'cluster' (int), 'core' (int) labeling each ping with their cluster id and core id or noise, indexed by 'unix_timestamp'
    """    
    if not neighbor_dict:
        neighbor_dict = find_neighbors(data, time_thresh, dist_thresh)
    else:
        valid_times = set(data.index)
        neighbor_dict = defaultdict(set, {k: v.intersection(valid_times) for k, v in neighbor_dict.items() if k in valid_times})
    
    cluster_df = pd.Series(-2, index=data.index, name='cluster')
    core_df = pd.Series(-1, index=data.index, name='core')
    
    cid = -1                                              # Initialize cluster label
    for i, cluster in cluster_df.items():
        if cluster < 0:                                   # Check if point is not yet in a cluster
            if len(neighbor_dict[i]) < min_pts:
                cluster_df[i] = -1                        # Mark as noise if below min_pts
            else:
                cid += 1
                cluster_df[i] = cid                       # Assign new cluster label
                core_df[i] = cid                          # Assign new core label
                S = list(neighbor_dict[i])                # Initialize stack with neighbors
                
                while S:
                    j = S.pop()
                    if cluster_df[j] < 0:                 # Process if not yet in a cluster
                        cluster_df[j] = cid
                        if len(neighbor_dict[j]) >= min_pts:
                            core_df[j] = cid              # Assign core label
                            for k in neighbor_dict[j]:
                                if cluster_df[k] < 0:
                                    S.append(k)           # Add new neighbors
    
    return pd.DataFrame({'cluster': cluster_df, 'core': core_df})

def process_clusters(data, time_thresh, dist_thresh, min_pts, output, cluster_df=None, neighbor_dict=None, min_duration=4):
    """
    TODO

    Parameters
    ----------
    data : pandas.DataFrame
        User pings with 'unix_timestamp' (integer), 'x' (EPSG:3857), 'y' (EPSG:3857) columns, indexed by 'unix_timestamp'
    time_thresh : int
        Time threshold in minutes.
    dist_thresh : float
        Distance threshold in meters.
    min_pts: int
        A cluster must have at least (min_pts+1) points to be considered a cluster.
    output: pandas.DataFrame
        TODO
    cluster_df : pandas.DataFrame
        Output of dbscan
    neighbor_dict: dictionary
        TODO 
    min_duration: int
        A cluster must have duration at least 'min_duration' to be considered a cluster.

    Returns
    -------
    List
        (start, duration, x_mean, y_mean, n, max_gap, radius) of each (post-processed) cluster
    """    
    if not neighbor_dict:
        neighbor_dict = find_neighbors(data, time_thresh, dist_thresh)
    if cluster_df is None:    # First call of process_clusters
        cluster_df = dbscan(data=data, time_thresh=time_thresh, dist_thresh=dist_thresh, min_pts=min_pts, neighbor_dict=neighbor_dict)
    if len(cluster_df) < min_pts:
        return False
        
    cluster_df = cluster_df[cluster_df['cluster'] != -1]    # Remove noise pings
    
    # All pings are in the same cluster
    if len(cluster_df['cluster'].unique()) == 1:
        x = dbscan(data=data.loc[cluster_df.index], time_thresh=time_thresh, dist_thresh=dist_thresh, min_pts=min_pts, neighbor_dict=neighbor_dict)   # We rerun dbscan because possibly these points no longer hold their own
        y = x.loc[x['cluster'] != -1] 
        z = x.loc[x['core'] != -1]
        
        # There is exactly 1 cluster of all the same values
        if len(y) > 0:
            duration = int((y.index.max() - y.index.min()) // 60)    # Assumes unix_timestamp is in seconds
            if duration > min_duration:
                cid = max(output['cluster']) + 1   # Create new cluster id
                output['cluster'].loc[y.index] = cid
                output['core'].loc[z.index] = cid
            return True
        elif len(y)==0:    # The points in df, despite originally being part of a cluster, no longer hold their own
            return False
        
    # There are no clusters
    elif len(cluster_df['cluster'].unique()) == 0:
        return False
    
    # There is more than one cluster
    else:
        i, j = extract_middle(cluster_df)    # Indices of the "middle" of the cluster (i.e., the head is the first contiguous cluster, and the middle follows that)
        # Recursively processes clusters
        if process_clusters(data, time_thresh, dist_thresh, min_pts, output, cluster_df[i:j]): # Valid cluster in the middle
            process_clusters(data, time_thresh, dist_thresh, min_pts, output, cluster_df[:i])  # Process the initial stub
            process_clusters(data, time_thresh, dist_thresh, min_pts, output, cluster_df[j:])  # Process the "tail"
            return True
        else: # No valid cluster in the middle
            return process_clusters(data, time_thresh, dist_thresh, min_pts, output, pd.concat( [cluster_df[:i],cluster_df[j:]] )) #what if this is out of bounds?
        
def temporal_dbscan(data, time_thresh, dist_thresh, min_pts):
    """
    TODO

    Parameters
    ----------
    data : pandas.DataFrame
        User pings with 'unix_timestamp' (integer), 'x' (EPSG:3857), 'y' (EPSG:3857) columns.
    time_thresh : int
        Time threshold in minutes.
    dist_thresh : float
        Distance threshold in meters.
    min_pts: int
        A cluster must have at least (min_pts+1) points to be considered a cluster.

    Returns
    -------
    TODO
    """    
    data = data.set_index('unix_timestamp', drop=False)
    output = pd.DataFrame({'cluster': -1, 'core': -1}, index=data.index)
    process_clusters(data=data, time_thresh=time_thresh, dist_thresh=dist_thresh, min_pts=min_pts, output=output, min_duration=4)

    return output