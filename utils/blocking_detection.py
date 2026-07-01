import numpy as np
import cv2
import xarray as xr
from scipy.optimize import linear_sum_assignment


def _list2arr(var_list, n_events, max_dur):
    arr = np.full((n_events, max_dur), 0, dtype=np.int32)
    for i, xx in enumerate(var_list):
        arr[i, :len(xx)] = xx
    return arr


def _label_periodic(mask, ny, nx):
    n, lbl, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for yy in range(ny):
        if mask[yy, 0] and mask[yy, -1]:
            union(lbl[yy, 0], lbl[yy, -1])

    new_id   = np.zeros(n, dtype=int)
    next_lab = 1
    for old in range(1, n):
        root = find(old)
        if new_id[root] == 0:
            new_id[root] = next_lab
            next_lab += 1
        new_id[old] = new_id[root]

    return new_id[lbl], next_lab - 1


def blocking_detection(var, var_a, var_c, time, ds, dx, dy, Ld, duration, size_factor, file_block, var_type, thresh_option):
    nday, ny, nx = var.shape
    x = np.arange(nx)
    y = np.arange(ny)

    if thresh_option == 0:
        lwa_max_x = np.array([np.max(var[t, :, lo]) for t in range(nday) for lo in range(nx)])
        thresh = np.median(lwa_max_x)
    elif thresh_option == 1:
        thresh = np.percentile(var, 99)

    WE = np.zeros((nday, ny, nx), dtype=np.uint8)
    WE[var > thresh] = 255

    num_labels = np.zeros(nday, dtype=int)
    labels     = np.zeros((nday, ny, nx), dtype=int)

    for d in range(nday):
        mask          = (WE[d] != 0).astype(np.uint8)
        labels[d], num_labels[d] = _label_periodic(mask, ny, nx)

    y_d = []; x_d = []
    x_w = []; x_e = []; area = []
    x_wide = []; lwa_we = []

    for d in range(nday):
        if int(num_labels[d]) == 0:
            nan_arr = np.array([np.nan])
            y_d.append(nan_arr); x_d.append(nan_arr)
            x_w.append(nan_arr); x_e.append(nan_arr)
            area.append(nan_arr); x_wide.append(nan_arr)
            lwa_we.append(nan_arr)
            continue

        n_ev      = int(num_labels[d])
        y_list    = np.zeros(n_ev); x_list    = np.zeros(n_ev)
        x_w_list  = np.zeros(n_ev); x_e_list  = np.zeros(n_ev)
        area_list = np.zeros(n_ev); x_wide_list = np.zeros(n_ev)
        max_list  = np.zeros(n_ev)

        for n in range(n_ev):
            WE_lwa = np.zeros((ny, nx))
            WE_lwa[labels[d] == n + 1] = var[d][labels[d] == n + 1]

            idx = np.array(np.where(WE_lwa == WE_lwa.max()))
            if idx.shape[1] > 1:
                y_list[n] = y[idx[0, 0]]
                x_list[n] = x[idx[1, 0]]
            else:
                y_list[n] = y[idx[0, 0]]
                x_list[n] = x[idx[1, 0]]
            max_list[n] = WE_lwa.max()

            for lo in range(nx):
                if np.any(WE_lwa[:, lo]) and not np.any(WE_lwa[:, lo - 1]):
                    x_w_list[n] = x[lo]
                if not np.any(WE_lwa[:, lo]) and np.any(WE_lwa[:, lo - 1]):
                    x_e_list[n] = x[lo - 1]

            if x_e_list[n] - x_w_list[n] > 0:
                x_wide_list[n] = x_e_list[n] - x_w_list[n]
            else:
                x_wide_list[n] = nx + (x_e_list[n] - x_w_list[n])

            area_list[n] = np.sum(ds[labels[d] == n + 1]) / 1e6

        y_d.append(y_list);   x_d.append(x_list)
        x_w.append(x_w_list); x_e.append(x_e_list)
        area.append(area_list); x_wide.append(x_wide_list)
        lwa_we.append(max_list)

    x_thresh = round(Ld / dx)
    y_thresh = round(Ld / dy)
    m_thresh = np.hypot(x_thresh * dx, y_thresh * dy)

    next_index = []
    for d in range(nday - 1):
        x1, y1 = np.array(x_d[d]),   np.array(y_d[d])
        x2, y2 = np.array(x_d[d+1]), np.array(y_d[d+1])
        n1     = len(x1)

        if n1 == 1 and np.isnan(x1[0]):
            next_index.append(np.array([]))
            continue
        if len(x2) == 1 and np.isnan(x2[0]):
            next_index.append(np.full(n1, np.nan))
            continue

        dx_mat = np.abs(x1[:, None] - x2[None, :])
        dx_mat = np.minimum(dx_mat, nx - dx_mat)
        dy_mat = np.abs(y1[:, None] - y2[None, :])
        dist   = np.hypot(dx_mat * dx, dy_mat * dy)

        row_ind, col_ind = linear_sum_assignment(dist)
        valid    = dist[row_ind, col_ind] <= m_thresh
        next_day = np.full(n1, np.nan)
        next_day[row_ind[valid]] = col_ind[valid]
        next_index.append(next_day)

    Blocking_y = [];    Blocking_x = [];     Blocking_date = []
    Blocking_x_wide = []; Blocking_area = []; Blocking_label = []; Blocking_lwa = []
    Blocking_peaking_y = []; Blocking_peaking_x = []
    Blocking_peaking_date = []; Blocking_peaking_lwa = []
    Blocking_duration = []; Blocking_peaking_label = []
    Blocking_type = []

    for d in range(nday - 1):
        if np.all(np.isnan(x_d[d])):
            continue

        for i in range(len(x_d[d])):
            if np.isnan(x_d[d][i]):
                continue

            day = 0
            track_x = []; track_y = []; track_date = []; track_lwa = []
            track_x_index = []; track_y_index = []
            track_x_wide = []; track_area = []; track_label = []

            track_x.append(x_d[d][i]);       track_x_index.append(i)
            track_y.append(y_d[d][i]);       track_y_index.append(i)
            track_date.append(time[d])
            track_x_wide.append(x_wide[d][i])
            track_area.append(area[d][i])
            track_label.append(labels[d] == i + 1)
            track_lwa.append(lwa_we[d][i])

            nip = next_index[d][i]
            if not np.isnan(nip):
                nip = int(nip)

            while not np.isnan(nip):
                track_date.append(time[d + day + 1])
                track_x.append(x_d[d + day + 1][nip])
                track_x_index.append(nip)
                track_y.append(y_d[d + day + 1][nip])
                track_y_index.append(nip)
                track_x_wide.append(x_wide[d + day + 1][nip])
                track_area.append(area[d + day + 1][nip])
                track_label.append(labels[d + day + 1] == nip + 1)
                track_lwa.append(lwa_we[d + day + 1][nip])
                day += 1

                if d + day + 1 > nday - 1:
                    break

                nip = next_index[d + day][nip]
                if not np.isnan(nip):
                    nip = int(nip)

            n_large = sum(1 for w in track_x_wide if w > round(size_factor * Ld / dx))

            if day + 1 >= duration and n_large >= duration:
                Blocking_x.append(track_x)
                Blocking_y.append(track_y)
                Blocking_date.append(track_date)
                Blocking_x_wide.append(track_x_wide)
                Blocking_area.append(track_area)
                Blocking_label.append(track_label)
                Blocking_lwa.append(track_lwa)
                Blocking_duration.append(day + 1)

                lwa_peak     = max(track_lwa)
                lwa_peak_idx = track_lwa.index(lwa_peak)
                Blocking_peaking_lwa.append(lwa_peak)
                Blocking_peaking_x.append(track_x[lwa_peak_idx])
                Blocking_peaking_y.append(track_y[lwa_peak_idx])
                Blocking_peaking_date.append(track_date[lwa_peak_idx])
                Blocking_peaking_label.append(track_label[lwa_peak_idx])

                peak_date  = track_date[lwa_peak_idx]
                time_index = np.where(time == peak_date)[0][0]
                Blocking_type.append(
                    blocking_type_v1(
                        var_a[time_index], var_c[time_index],
                        track_label[lwa_peak_idx], track_x[lwa_peak_idx],
                        ny, nx, Ld, dx, size_factor
                    )
                )

            for dd in range(day + 1):
                x_d[d + dd][track_x_index[dd]] = np.nan
                y_d[d + dd][track_y_index[dd]] = np.nan

        print(d)

    n_events = len(Blocking_x)
    max_dur  = max(len(xx) for xx in Blocking_x)

    ds_out = xr.Dataset(
        data_vars={
            "Blocking_x":             (("event", "time"), _list2arr(Blocking_x,      n_events, max_dur)),
            "Blocking_y":             (("event", "time"), _list2arr(Blocking_y,      n_events, max_dur)),
            "Blocking_date":          (("event", "time"), _list2arr(Blocking_date,   n_events, max_dur)),
            "Blocking_area":          (("event", "time"), _list2arr(Blocking_area,   n_events, max_dur)),
            "Blocking_x_wide":        (("event", "time"), _list2arr(Blocking_x_wide, n_events, max_dur)),
            "Blocking_lwa":           (("event", "time"), _list2arr(Blocking_lwa,    n_events, max_dur)),
            "Blocking_peaking_x":     (("event",),        np.array(Blocking_peaking_x)),
            "Blocking_peaking_y":     (("event",),        np.array(Blocking_peaking_y)),
            "Blocking_peaking_lwa":   (("event",),        np.array(Blocking_peaking_lwa)),
            "Blocking_peaking_date":  (("event",),        np.array(Blocking_peaking_date)),
            "Blocking_duration":      (("event",),        np.array(Blocking_duration)),
            "Blocking_peaking_label": (("event", "y", "x"), np.array(Blocking_peaking_label)),
            "Blocking_type":          (("event",),        np.array(Blocking_type)),
        },
        coords={
            "event": np.arange(n_events),
            "time":  np.arange(max_dur),
            "y":     np.arange(ny),
            "x":     np.arange(nx),
        },
    )
    ds_out.to_netcdf(file_block)


def blocking_type_v1(lwa_a_peak, lwa_c_peak, peaking_labels, peaking_x_index, ny, nx, Ld, dx, size_factor):
    x_range = round(size_factor * Ld / dx)
    shift   = int(nx / 2) - int(peaking_x_index)

    lwa_a_roll   = np.roll(lwa_a_peak,    shift, axis=1)
    lwa_c_roll   = np.roll(lwa_c_peak,    shift, axis=1)
    labels_roll  = np.roll(peaking_labels, shift, axis=1)

    cx = int(nx / 2)
    lwa_a_core  = lwa_a_roll[:,  cx - x_range : cx + x_range + 1]
    lwa_c_core  = lwa_c_roll[:,  cx - x_range : cx + x_range + 1]
    labels_core = labels_roll[:, cx - x_range : cx + x_range + 1]

    lwa_a_block = np.zeros((ny, 2 * x_range + 1))
    lwa_c_block = np.zeros((ny, 2 * x_range + 1))
    lwa_a_block[labels_core] = lwa_a_core[labels_core]
    lwa_c_block[labels_core] = lwa_c_core[labels_core]

    lwa_a_sum = lwa_a_block.sum()
    lwa_c_sum = lwa_c_block.sum()

    if lwa_a_sum > 5 * lwa_c_sum:
        return 0
    elif lwa_c_sum > 5 * lwa_a_sum:
        return 1
    return 2


def blocking_type_v2(lwa_a_all, lwa_c_all, labels_all, x_all, ny, nx, Ld, dx, size_factor):
    x_range   = round(size_factor * Ld / dx)
    lwa_a_sum = 0
    lwa_c_sum = 0

    for d in range(len(lwa_a_all)):
        shift = int(nx / 2) - int(x_all[d])

        lwa_a_roll  = np.roll(lwa_a_all[d],  shift, axis=1)
        lwa_c_roll  = np.roll(lwa_c_all[d],  shift, axis=1)
        labels_roll = np.roll(labels_all[d], shift, axis=1)

        cx = int(nx / 2)
        lwa_a_core  = lwa_a_roll[:,  cx - x_range : cx + x_range + 1]
        lwa_c_core  = lwa_c_roll[:,  cx - x_range : cx + x_range + 1]
        labels_core = labels_roll[:, cx - x_range : cx + x_range + 1]

        lwa_a_block = np.zeros((ny, 2 * x_range + 1))
        lwa_c_block = np.zeros((ny, 2 * x_range + 1))
        lwa_a_block[labels_core] = lwa_a_core[labels_core]
        lwa_c_block[labels_core] = lwa_c_core[labels_core]

        lwa_a_sum += lwa_a_block.sum()
        lwa_c_sum += lwa_c_block.sum()

    if lwa_a_sum > 2 * lwa_c_sum:
        return 0
    elif lwa_c_sum > 2 * lwa_a_sum:
        return 1
    return 2