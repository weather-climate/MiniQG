import numpy as np
import pandas as pd


def reconstruct_float_from_sequence(int_values, start_value, increment):
    reconstructed_floats = np.zeros_like(int_values, dtype=float)

    for i, int_val in enumerate(int_values):
        if int_val == 0:
            continue

        n_min = (int_val - start_value) / increment
        n_max = (int_val + 1 - start_value) / increment
        n = int(np.ceil(n_min))

        if n >= n_max:
            n = n - 1

        candidate_float = start_value + n * increment

        if int(np.floor(candidate_float)) == int_val:
            reconstructed_floats[i] = candidate_float
        else:
            for n_try in [n - 1, n + 1]:
                candidate_float = start_value + n_try * increment
                if int(np.floor(candidate_float)) == int_val:
                    reconstructed_floats[i] = candidate_float
                    break

    return reconstructed_floats


def rescale_to_target_range(float_values, original_min, original_max, target_min, target_max):
    non_zero_mask = float_values != 0
    rescaled = np.zeros_like(float_values, dtype=int)

    if np.any(non_zero_mask):
        non_zero_values = float_values[non_zero_mask]
        rescaled_values = target_min + (non_zero_values - original_min) * (target_max - target_min) / (original_max - original_min)
        rescaled[non_zero_mask] = np.round(rescaled_values).astype(int)

    return rescaled


def create_rescaled_dataset(ds, start_value, increment, original_time_min, original_time_max, target_min, target_max):
    new_ds = ds.copy(deep=True)

    if "Blocking_date" in ds.data_vars:
        var_data = ds["Blocking_date"]
        var_name = "Blocking_date"

        if var_data.dtype in [np.int32, np.int64] and 'time' in var_data.dims:
            if var_data.ndim == 2 and 'event' in var_data.dims and 'time' in var_data.dims:
                rescaled_data = np.zeros_like(var_data.values)

                for event_idx in range(var_data.sizes['event']):
                    int_sequence = var_data.isel(event=event_idx).values

                    if np.all(int_sequence == 0):
                        continue

                    reconstructed_floats = reconstruct_float_from_sequence(int_sequence, start_value, increment)
                    rescaled_sequence = rescale_to_target_range(
                        reconstructed_floats, original_time_min, original_time_max, target_min, target_max
                    )
                    rescaled_data[event_idx] = rescaled_sequence

                new_ds[var_name] = (var_data.dims, rescaled_data.astype(var_data.dtype))

    return new_ds


def analyze_blocking_events(ds, a, b, c, d):
    blocking_data = ds['Blocking_date'].values

    spans = []
    event_date_values = []

    for event_idx in range(blocking_data.shape[0]):
        event_data = blocking_data[event_idx, :]
        non_zero_mask = event_data != 0
        spans.append(np.sum(non_zero_mask))
        event_date_values.append(event_data[non_zero_mask])

    spans = np.array(spans)
    unique_spans = np.unique(spans)
    unique_spans = unique_spans[unique_spans > 0]

    results = []

    cumulative_ab = 0
    cumulative_ab_bc = 0
    cumulative_bc = 0
    cumulative_bc_cd = 0
    cumulative_cd = 0
    cumulative_days = set()
    cumulative_days_ab = set()
    cumulative_days_ab_bc = set()
    cumulative_days_bc = set()
    cumulative_days_bc_cd = set()
    cumulative_days_cd = set()

    for span in sorted(unique_spans, reverse=True):
        events_with_span = np.sum(spans == span)
        event_indices = np.where(spans == span)[0]

        all_date_values = []
        for event_idx in event_indices:
            all_date_values.extend(event_date_values[event_idx])
        all_date_values = np.array(all_date_values)

        events_ab = events_ab_bc = events_bc = events_bc_cd = events_cd = 0
        days_ab_events = set()
        days_ab_bc_events = set()
        days_bc_events = set()
        days_bc_cd_events = set()
        days_cd_events = set()

        for event_idx in event_indices:
            date_vals = event_date_values[event_idx]
            min_date = np.min(date_vals)
            max_date = np.max(date_vals)

            in_ab = (min_date >= a) and (max_date <= b)
            in_bc = (min_date >= b) and (max_date <= c)
            in_cd = (min_date >= c) and (max_date <= d)
            spans_ab_bc = (min_date >= a) and (min_date < b) and (max_date > b) and (max_date <= c)
            spans_bc_cd = (min_date >= b) and (min_date < c) and (max_date > c) and (max_date <= d)

            if in_ab:
                events_ab += 1
                days_ab_events.update(date_vals)
            if spans_ab_bc:
                events_ab_bc += 1
                days_ab_bc_events.update(date_vals)
            if in_bc:
                events_bc += 1
                days_bc_events.update(date_vals)
            if spans_bc_cd:
                events_bc_cd += 1
                days_bc_cd_events.update(date_vals)
            if in_cd:
                events_cd += 1
                days_cd_events.update(date_vals)

        cumulative_ab += events_ab
        cumulative_ab_bc += events_ab_bc
        cumulative_bc += events_bc
        cumulative_bc_cd += events_bc_cd
        cumulative_cd += events_cd

        cumulative_days.update(all_date_values)
        cumulative_days_ab.update(days_ab_events)
        cumulative_days_ab_bc.update(days_ab_bc_events)
        cumulative_days_bc.update(days_bc_events)
        cumulative_days_bc_cd.update(days_bc_cd_events)
        cumulative_days_cd.update(days_cd_events)

        results.append({
            'span': span,
            'events': events_with_span,
            'events_tup': (events_ab, events_ab_bc, events_bc, events_bc_cd, events_cd),
            'cumu_events': int(np.sum(spans >= span)),
            'cumu_events_tup': (cumulative_ab, cumulative_ab_bc, cumulative_bc, cumulative_bc_cd, cumulative_cd),
            'days': len(set(all_date_values)),
            'days_tup': (len(days_ab_events), len(days_ab_bc_events), len(days_bc_events), len(days_bc_cd_events), len(days_cd_events)),
            'cumu_days': len(cumulative_days),
            'cumu_days_tup': (len(cumulative_days_ab), len(cumulative_days_ab_bc),
                              len(cumulative_days_bc), len(cumulative_days_bc_cd),
                              len(cumulative_days_cd))
        })

    return results


def get_events_in_range(ds, min_val, max_val):
    blocking_data = ds['Blocking_date'].values
    found = []

    for event_idx in range(blocking_data.shape[0]):
        event_data = blocking_data[event_idx, :]
        non_zero_mask = event_data != 0

        if np.sum(non_zero_mask) == 0:
            continue

        date_vals = event_data[non_zero_mask]
        if (np.min(date_vals) >= min_val) and (np.max(date_vals) <= max_val):
            found.append({
                'event_idx': event_idx,
                'span': len(date_vals),
                'dates': [int(x) for x in sorted(date_vals)]
            })

    return found


def get_blocking_days(ds, threshold, time_range, train_ratio=0.8):
    start_day, end_day = time_range
    train_end = start_day + int((end_day - start_day) * train_ratio)

    blocking_days = set()

    for event_id in range(ds.sizes['event']):
        dates = ds["Blocking_date"].isel(event=event_id).values
        nonzero_dates = sorted(set(int(d) for d in dates if d > 0))

        if not nonzero_dates:
            continue

        span = nonzero_dates[-1] - nonzero_dates[0] + 1

        if span >= threshold:
            for day in nonzero_dates:
                if start_day <= day < train_end:
                    blocking_days.add(day)

    return sorted(blocking_days)


def get_events_spanning_threshold(ds, threshold):
    blocking_data = ds['Blocking_date'].values
    found = []

    for event_idx in range(blocking_data.shape[0]):
        event_data = blocking_data[event_idx, :]
        non_zero_mask = event_data != 0

        if np.sum(non_zero_mask) == 0:
            continue

        date_vals = event_data[non_zero_mask]
        if np.any(date_vals < threshold) and np.any(date_vals >= threshold):
            found.append({
                'event_idx': event_idx,
                'span': len(date_vals),
                'dates_before': [int(x) for x in sorted(date_vals[date_vals < threshold])],
                'dates_after': [int(x) for x in sorted(date_vals[date_vals >= threshold])],
            })

    return found