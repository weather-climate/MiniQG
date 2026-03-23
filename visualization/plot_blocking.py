import numpy as np
import matplotlib.pyplot as plt


def plot_blocking_trajectory(ds, event_id, cmap='tab20'):
    x_points = ds["Blocking_x"].isel(event=event_id).values
    y_points = ds["Blocking_y"].isel(event=event_id).values
    t_steps = np.arange(len(x_points))

    nlat = ds.dims.get('y', int(np.nanmax(y_points)) + 1)
    nlon = ds.dims.get('x', int(np.nanmax(x_points)) + 1)

    plt.figure(figsize=(6, 6))
    plt.imshow(np.zeros((nlat, nlon)), cmap='gray_r', origin='lower',
               extent=[0, nlon, 0, nlat])
    sc = plt.scatter(x_points, y_points, c=t_steps, cmap=cmap,
                     marker='o', s=40, edgecolors='k')
    plt.title(f'Blocking Event {event_id} (trajectory)')
    plt.xlabel('Longitude index (x)')
    plt.ylabel('Latitude index (y)')
    plt.xlim(0, nlon)
    plt.ylim(0, nlat)
    plt.colorbar(sc, label='time step')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.show()


def plot_blocking_trajectories(ds, event_ids, cmap='tab20'):
    for event_id in event_ids:
        plot_blocking_trajectory(ds, event_id, cmap=cmap)