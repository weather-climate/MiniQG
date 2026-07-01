import numpy as np


def LWA(qgpv, phi, Nx, Ny, dx, dy, ds):
    x  = np.arange(Nx)
    y  = np.arange(Ny)
    _, yy = np.meshgrid(x, y)

    qref = eqlat_q(qgpv, Nx, Ny, y, dx, dy, ds)
    zref = eqlat_z(phi,  Nx, Ny, y, dx, dy, ds)

    lwa, lwa_a, lwa_c, fawa, fawa_a, fawa_c = integrate_q(qgpv, qref, Ny, Nx, yy, dx, dy, ds)
    lwa_z, lwa_z_a, lwa_z_c                 = integrate_z(phi,  zref, Ny, Nx, yy, dx, dy, ds)

    return lwa, lwa_a, lwa_c, qref, zref, lwa_z, lwa_z_a, lwa_z_c


def eqlat_q(qgpv, Nx, Ny, y, dx, dy, ds):
    npart = Ny
    yi_t  = np.zeros(npart)
    levs  = np.linspace(qgpv.min(), qgpv.max(), npart)

    for j in range(npart):
        A       = np.sum(ds[qgpv >= levs[j]])
        yi_t[j] = (Ny * dy - (A / (Nx * dx))) / dy

    return np.interp(y, yi_t, levs)


def eqlat_z(phi, Nx, Ny, y, dx, dy, ds):
    npart = Ny
    yi_t  = np.zeros(npart)
    levs  = np.linspace(phi.min(), phi.max(), npart)

    for j in range(npart):
        A             = np.sum(ds[phi <= levs[j]])
        yi_t[Ny-1-j]  = (Ny * dy - (A / (Nx * dx))) / dy

    return np.interp(y, yi_t, levs[::-1])


def integrate_q(qgpv, qref, Ny, Nx, yy, dx, dy, ds):
    y = np.arange(Ny)

    lwa    = np.zeros((Ny, Nx))
    lwa_a  = np.zeros((Ny, Nx))
    lwa_c  = np.zeros((Ny, Nx))
    fawa   = np.zeros(Ny)
    fawa_a = np.zeros(Ny)
    fawa_c = np.zeros(Ny)

    for j in range(Ny):
        qe   = qgpv - qref[j]
        qb   = np.zeros((Ny, Nx))
        qb_a = np.zeros((Ny, Nx))
        qb_c = np.zeros((Ny, Nx))

        qb[(yy <= y[j]) & (qe >= 0)] =  1
        qb[(yy >= y[j]) & (qe <= 0)] = -1
        lwa[j,:]  = np.sum(qb * qe * dy, axis=0)
        fawa[j]   = np.sum(qb * qe * ds) / (Nx * dx)

        qb_a[(yy >= y[j]) & (qe <= 0)] = -1
        lwa_a[j,:]  = np.sum(qb_a * qe * dy, axis=0)
        fawa_a[j]   = np.sum(qb_a * qe * ds) / (Nx * dx)

        qb_c[(yy <= y[j]) & (qe >= 0)] = 1
        lwa_c[j,:]  = np.sum(qb_c * qe * dy, axis=0)
        fawa_c[j]   = np.sum(qb_c * qe * ds) / (Nx * dx)

    return lwa, lwa_a, lwa_c, fawa, fawa_a, fawa_c


def integrate_z(phi, zref, Ny, Nx, yy, dx, dy, ds):
    y = np.arange(Ny)

    lwa_z   = np.zeros((Ny, Nx))
    lwa_z_a = np.zeros((Ny, Nx))
    lwa_z_c = np.zeros((Ny, Nx))

    for j in range(Ny):
        ze   = phi - zref[j]
        zb   = np.zeros((Ny, Nx))
        zb_a = np.zeros((Ny, Nx))
        zb_c = np.zeros((Ny, Nx))

        zb[(yy >= y[j]) & (ze >= 0)] =  1
        zb[(yy <= y[j]) & (ze <= 0)] = -1
        lwa_z[j,:]   = np.sum(zb * ze * dy, axis=0)

        zb_a[(yy >= y[j]) & (ze >= 0)] = 1
        lwa_z_a[j,:] = np.sum(zb_a * ze * dy, axis=0)

        zb_c[(yy <= y[j]) & (ze <= 0)] = -1
        lwa_z_c[j,:] = np.sum(zb_c * ze * dy, axis=0)

    return lwa_z, lwa_z_a, lwa_z_c


def dAdt_diabatic_cal(qgpv, qref, dqdt_diabatic, Nx, Ny, dx, dy, ds):
    x  = np.arange(Nx)
    y  = np.arange(Ny)
    _, yy = np.meshgrid(x, y)

    dAdt_diabatic   = np.zeros((Ny, Nx))
    dAdt_diabatic_a = np.zeros((Ny, Nx))
    dAdt_diabatic_c = np.zeros((Ny, Nx))
    fawa_diabatic   = np.zeros(Ny)
    fawa_diabatic_a = np.zeros(Ny)
    fawa_diabatic_c = np.zeros(Ny)

    for j in range(Ny):
        qe   = qgpv - qref[j]
        qb   = np.zeros((Ny, Nx))
        qb_a = np.zeros((Ny, Nx))
        qb_c = np.zeros((Ny, Nx))

        qb[(yy <= y[j]) & (qe >= 0)] =  1
        qb[(yy >= y[j]) & (qe <= 0)] = -1
        dAdt_diabatic[j,:]  = np.sum(qb * dqdt_diabatic * dy, axis=0)
        fawa_diabatic[j]    = np.sum(qb * dqdt_diabatic * ds) / (Nx * dx)

        qb_a[(yy >= y[j]) & (qe <= 0)] = -1
        dAdt_diabatic_a[j,:] = np.sum(qb_a * dqdt_diabatic * dy, axis=0)
        fawa_diabatic_a[j]   = np.sum(qb_a * dqdt_diabatic * ds) / (Nx * dx)

        qb_c[(yy <= y[j]) & (qe >= 0)] = 1
        dAdt_diabatic_c[j,:] = np.sum(qb_c * dqdt_diabatic * dy, axis=0)
        fawa_diabatic_c[j]   = np.sum(qb_c * dqdt_diabatic * ds) / (Nx * dx)

    return dAdt_diabatic, dAdt_diabatic_a, dAdt_diabatic_c