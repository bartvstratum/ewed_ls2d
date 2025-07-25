from datetime import datetime
import json
import glob
import os

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.interpolate import interp1d
import pandas as pd
import numpy as np
import ls2d

# Constants.
Rd = 287.05
cp = 1005.
p0 = 1e5
T0 = 273.15


def get_web_dict():
    """
    Create dictionary with data structure and variable
    names used by web interface. Some reasonable first
    values are set, which require manual tuning.
    """
    return dict(
        reference = dict(
            name = 'case_name',
            description = 'default description',
            h = 500,
            theta = 290,
            dtheta = 2,
            gamma_theta = [0.003, 0.006],
            z_theta =[1500, 3720],
            advtheta = 0,
            qt = 10e-3,
            dqt = -2e-3,
            gamma_qt = [-0.002e-3, -0.003e-3],
            z_qt = [1200, 3720],
            advq = 0,
            divU = 1e-5,
            u = 5,
            ug = 7,
            du = 2,
            gamma_u = [1e-3, 1e-3],
            z_u = [1200, 3720],
            advu = 0,
            v = 5,
            vg = 6,
            dv = 1,
            gamma_v = [-1e-3, -1e-3],
            z_v = [1200, 3720],
            advv = 0,
            ustar = 0.1,
            runtime = 10800,
            wtheta = [0],
            wq = [0],
            fc = 0,
            p0 = 101300,
            z0m = 0,
            z0h = 0,
            is_tuned = False
            ),
        preset = 'case_name',
        permutations = []
    )


#def get_mxl_dict():
#    """
#    Create dictionary with all mixed-layer properties:
#    Initial profiles, forcings, surface fluxes, ...
#    Initial values are completely (well, almost) random.
#    """
#    return dict(
#        zi = 500,
#        thl = 290,
#        dthl = 2,
#        gamma_thl_1 = 0.003,
#        z1_thl = 1500,
#        gamma_thl_2 = 0.006,
#        qt = 10e-3,
#        dqt = -2e-3,
#        gamma_qt_1 = -0.002e-3,
#        z1_qt = 1200,
#        gamma_qt_2 = -0.003e-3,
#        div = 1e-5,
#        u = 5,
#        ug = 7,
#        du = 2,
#        gamma_u_1 = 1e-3,
#        z1_u = 1200,
#        gamma_u_2 = 1e-3,
#        v = 5,
#        vg = 6,
#        dv = 1,
#        gamma_v_1 = -1e-3,
#        z1_v = 1200,
#        gamma_v_2 = -1e-3,
#        start_time = datetime(1900,1,1).isoformat(),
#        time = [0],
#        wthl = [0],
#        wqt = [0],
#        fc = 0,
#        ps = 0,
#        z0m = 0,
#        z0h = 0,
#        is_tuned = False
#    )


def load_json(json_file):
    """
    Load `json_file` from disk into dict.
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data, json_file):
    """
    Save `data` to `json_file`.
    """
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


class Mixed_layer_profile:
    def __init__(self, fig, ax, x, y, instances):
        """
        Interactive / dragable mixed-layer profile.
        """
        self.fig = fig
        self.ax = ax
        self.x = np.array(x, dtype=np.float32)
        self.y = np.array(y, dtype=np.float32)
        self.instances = instances
        
        self.points, = ax.plot(self.x, self.y, 'o', color='b', mfc='none', picker=True, pickradius=5)
        self.line,   = ax.plot(self.x, self.y, 'b--')

        self.labels = ['bulk', 'jump', 'zi', 'z1', 'gamma1', 'gamma2']
        self.texts = []
        for label in self.labels:
            x, y, text, ha, va = self.get_label(label)
            self.texts.append(plt.text(x, y, text, ha=ha, va=va))

        self.index = None

        self.cid_press = fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = fig.canvas.mpl_connect('motion_notify_event', self.on_motion)


    def get_label(self, name):
        """
        Get label incl. coordinates and alignments.
        """

        x_margin = 0.04 * (self.x.max() - self.x.min())
        y_margin = 0.02 * (self.y.max() - self.y.min())

        sign = +1 if self.x[2] > self.x[1] else -1

        if name == 'bulk':
            x = self.x[0] + x_margin
            y = 0.5 * (self.y[0] + self.y[1])
            text = f'<φ>={self.x[0]:.1f}'
            ha = 'left'
            va = 'center'
        
        elif name == 'zi':
            x = self.x[0] - sign * x_margin
            y = self.y[1]
            text = rf'$z_i$={self.y[1]:.0f}'
            ha = 'right' if sign > 0 else 'left'
            va = 'center'

        elif name == 'jump':
            x = 0.5 * (self.x[1] + self.x[2])
            y = self.y[1] + y_margin
            text = f'Δ={self.x[2]-self.x[1]:.1f}'
            ha = 'center'
            va = 'bottom'

        elif name == 'z1':
            x = self.x[3] + x_margin
            y = self.y[3]
            text = rf'$z_1$={self.y[3]:.0f}'
            ha = 'left'
            va = 'center'

        elif name == 'gamma1':
            x = 0.5 * (self.x[2] + self.x[3]) + x_margin
            y = 0.5 * (self.y[2] + self.y[3])
            text = rf'$γ_1$={(self.x[3] - self.x[2])/(self.y[3] - self.y[2]) * 1000:.1f}'
            ha = 'left'
            va = 'center'

        elif name == 'gamma2':
            x = 0.5 * (self.x[3] + self.x[4]) + x_margin
            y = 0.5 * (self.y[3] + self.y[4])
            text = rf'$γ_2$={(self.x[4] - self.x[3])/(self.y[4] - self.y[3]) * 1000:.1f}'
            ha = 'left'
            va = 'center'

        return x, y, text, ha, va


    def on_press(self, event):

        if event.inaxes != self.ax:
            return

        x_lim = self.ax.get_xlim()
        y_lim = self.ax.get_ylim()

        x_range = x_lim[1] - x_lim[0]
        y_range = y_lim[1] - y_lim[0]

        x_norm = (self.x - x_lim[0]) / x_range
        y_norm = (self.y - y_lim[0]) / y_range
    
        event_x_norm = (event.xdata - x_lim[0]) / x_range
        event_y_norm = (event.ydata - y_lim[0]) / y_range

        distances = np.hypot(x_norm - event_x_norm, y_norm - event_y_norm)
    
        closest_index = np.argmin(distances)
        if distances[closest_index] < 0.05:
            self.index = closest_index


    def on_release(self, event):
        self.index = None


    def on_motion(self, event):
        if self.index is None or event.inaxes != self.ax:
            return

        # Some limits for mixed-layer shaped profile.
        # 1. Mixed-layer should stay mixed.
        if self.index == 0 or self.index == 1:
            self.x[0] = event.xdata
            self.x[1] = event.xdata

            self.y[self.index] = event.ydata

        # 2. Inversion should stay level.
        if self.index == 1 or self.index == 2:

            # Update height in all `Mixed_layer_profile` instances.
            for instance in self.instances:
                
                if instance == self:
                    instance.x[self.index] = event.xdata

                instance.y[1] = event.ydata
                instance.y[2] = event.ydata
        
        # Only update value, not height, for topmost point.
        if self.index == 4:
            self.x[self.index] = event.xdata

        # 3. Other points are free.
        elif self.index > 2:
            self.x[self.index] = event.xdata
            self.y[self.index] = event.ydata

        # 4. Make sure first point stays at z=0.
        if self.index == 0:
            self.y[0] = 0.

        # Update all profiles, including sister ones.
        for instance in self.instances:
            instance.update_plot()
    

    def update_plot(self):

        self.points.set_data(self.x, self.y)
        self.line.set_data(self.x, self.y)

        # Update all texts.
        for i, label in enumerate(self.labels):
            x, y, text, _, _ = self.get_label(label)
            self.texts[i].set_position((x, y))
            self.texts[i].set_text(text)

        #self.fig.canvas.blit(self.ax.bbox)  # Faster rendering
        #self.fig.canvas.draw()
        self.fig.canvas.draw_idle()


class Subsidence_profile:
    def __init__(self, fig, ax, div_U):
        self.fig = fig
        self.ax = ax

        self.div_U = div_U

        self.y = np.array([0, 4000], dtype=np.float32)
        self.x = (-div_U * self.y).astype(np.float32)
        
        self.points, = ax.plot(self.x, self.y, 'o', color='b', mfc='none', picker=True, pickradius=5)
        self.line,   = ax.plot(self.x, self.y, 'b--')

        x_margin = 0.04 * (self.x.max() - self.x.min())
        self.text = plt.text(self.x.mean()+x_margin, self.y.mean(), rf'div(U)={self.div_U:.1e}', ha='left', va='center')

        self.index = None

        self.cid_press = fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = fig.canvas.mpl_connect('motion_notify_event', self.on_motion)


    def on_press(self, event):

        if event.inaxes != self.ax:
            return

        self.index = 1


    def on_release(self, event):
        self.index = None


    def on_motion(self, event):

        if self.index is None or event.inaxes != self.ax:
            return

        self.x[1] = event.xdata

        self.points.set_data(self.x, self.y)
        self.line.set_data(self.x, self.y)

        # div(u) = -wls / z
        self.div_U = -self.x[1] / self.y[1]

        x_margin = 0.04 * (self.x.max() - self.x.min())
        self.text.set_position((self.x.mean()+x_margin, self.y.mean()))
        self.text.set_text(rf'div(U)={self.div_U:.1e}')

        #self.fig.canvas.blit(self.ax.bbox)  # Faster rendering
        #self.fig.canvas.draw()
        self.fig.canvas.draw_idle()


def parse_sounding(sounding_csv):
    """
    Read single sounding into Pandas dataframe,
    and calculate derived properties like potential
    temperature, specific humidity, et cetera.
    """
    def esat(T):
        return 0.611e3 * np.exp(17.2694 * (T - 273.16) / (T - 35.86))

    def qsat(T,p):
        return 0.622 * esat(T) / p

    # Parse CSV with Pandas.
    df = pd.read_csv(sounding_csv, parse_dates=['timestamp'], index_col=['timestamp'])

    # To SI...
    df['temperature'] += T0  # Celsius to Kelvin
    #df['speed'] /= 3.6  # km/h to m/s

    # Calculate derived properties.
    # Absolute to potential temperature.
    df['exner'] = (df['pressure'] / p0)**(Rd/cp)
    df['theta'] = df['temperature'] / df['exner']

    # Relative to specific humidity.
    es = esat(df['temperature'])
    e = df['relative_humidity'] / 100 * es
    df['qt'] = e * 0.622 / df['pressure']

    # Wind speed + direction to components.
    # NOTE: heading is NOT following the meteorological convention, but 180 degrees off.
    wind_dir_rad = np.deg2rad(df['heading'])
    df['u'] = df['speed'] * np.sin(wind_dir_rad)
    df['v'] = df['speed'] * np.cos(wind_dir_rad)

    return df


def interpolate_sounding_to_height(df, target_agl, agl_col='agl', kind='linear'):
    """
    Interpolates all columns in a DataFrame (except the height column)
    to a fixed set of height levels, grouped by time.
    """
    # Average duplicate heights.
    df = df.groupby(agl_col, as_index=False).mean()

    df_sorted = df.sort_values(agl_col)
    x = df_sorted[agl_col].values
    target_agl = np.asarray(target_agl)

    result = {agl_col: target_agl}
    for col in df.columns:
        if col == agl_col:
            continue
        y = df_sorted[col].values
        f = interp1d(x, y, kind=kind, fill_value='extrapolate', bounds_error=False)
        result[col] = f(target_agl)

    df = pd.DataFrame(result).set_index(agl_col)

    # Remove NaN values.
    df = df.interpolate(method='linear', axis=0, limit_direction='both')

    return df


class Fire_case:
    def __init__(self, name, lon, lat, start, end, ls2d_path):
        """
        Class to download / parse a single wild fire case.
        """
        self.name = name

        self.lon = lon
        self.lat = lat

        self.start = datetime.fromisoformat(start)
        self.end = datetime.fromisoformat(end)

        self.ls2d_settings = {
            'central_lat' : self.lat,
            'central_lon' : self.lon,
            'area_size'   : 1,
            'case_name'   : self.name,
            'era5_path'   : ls2d_path,
            'era5_expver' : 1,
            'start_date'  : self.start,
            'end_date'    : self.end,
            'write_log'   : False,
            'data_source' : 'CDS'}

        # Read or create dictionary with mixed-layer structure.
        self.json_file = f'json/{name}_web.json'

        if os.path.exists(self.json_file):
            print(f'Found tuned case for \"{name}\", loading...')
            self.mxl = load_json(self.json_file)
        else:
            print(f'No tuning for \"{name}\", creating new one...')
            self.mxl = get_web_dict()

        # Group with the case parameters.
        self.mxl_ref = self.mxl['reference']

        # Read soundings.
        sounding_csv = glob.glob(f'soundings/{name}/*.csv')
        self.soundings = {}

        if len(sounding_csv) > 0:
            print(f'Found {len(sounding_csv)} sounding(s) for \"{name}\".')

            for csv in sounding_csv:
                self.soundings[csv] = parse_sounding(csv)


    def download(self):
        """
        Download required ERA5 data using (LS)2D.
        """
        ls2d.download_era5(self.ls2d_settings, exit_when_waiting=False)


    def read(self):
        """
        Read ERA5 data with (LS)2D, and calculate large-scale forcings.
        """
        _era5 = ls2d.Read_era5(self.ls2d_settings)
        _era5.calculate_forcings(n_av=1, method='2nd')

        # Interpolate ERA5 to fixed height grid:
        self.z = np.arange(10., 4000., 20.)
        self.era5 = _era5.get_les_input(self.z)

        # Remove negative surface fluxes.
        # Does not work in a mixed-layer model, as this results in negative BL growth.
        self.era5.wth[self.era5.wth < 0] = 0
        self.era5.wq [self.era5.wq  < 0] = 0


    def plot(self):
        """
        Plot ERA5 initial profiles.
        """

        def get_range(time_prof):
            min_val = np.min(time_prof, axis=0)
            max_val = np.max(time_prof, axis=0)
            mean_val = np.mean(time_prof, axis=0)
            return min_val, mean_val, max_val

        def fmt_time():
            ax = plt.gca()
            hh = mdates.HourLocator(interval=3)
            fmt = mdates.DateFormatter('%H:%M')
            ax.xaxis.set_major_locator(hh)
            ax.xaxis.set_major_formatter(fmt)

        # For new input, guess mixed-layer values.
        if not self.mxl_ref['is_tuned']:

            self.mxl_ref['theta'] = self.era5.thl[0,:10].mean()
            self.mxl_ref['qt']  = self.era5.qt[0,:10].mean()

            self.mxl_ref['u'] = self.era5.u[0,:10].mean()
            self.mxl_ref['v']  = self.era5.v[0,:10].mean()

        def get_mxl_prof(var):
            """
            Create mixed-layer profile.
            """
            z0 = self.mxl_ref['h']
            z1 = self.mxl_ref[f'z_{var}'][0]
            z2 = self.mxl_ref[f'z_{var}'][1]

            v0 = self.mxl_ref[f'{var}']
            v1 = v0 + self.mxl_ref[f'd{var}']
            v2 = v1 + (z1-z0) * self.mxl_ref[f'gamma_{var}'][0]
            v3 = v2 + (z2-z1) * self.mxl_ref[f'gamma_{var}'][1]

            z = np.array([0, z0, z0, z1, z2])
            p = np.array([v0, v0, v1, v2, v3])

            if var == 'qt':
                p *= 1000.

            return z, p


        def plot_soundings(var, scale_fac=1):
            """
            Plot soundings for variable `var`.
            """
            if len(self.soundings) == 0:
                return

            for name, sounding in self.soundings.items():
                plt.plot(sounding[var]*scale_fac, sounding['agl'], dashes=[2,1], label=f'Sounding {sounding.index[0]} UTC')


        z_thl, p_thl = get_mxl_prof('theta')
        z_qt, p_qt = get_mxl_prof('qt')
        z_u, p_u = get_mxl_prof('u')
        z_v, p_v = get_mxl_prof('v')

        z_top = 4000

        self.mxl_profs = []

        fig=plt.figure(figsize=(10,8), layout='tight')
        plt.suptitle(rf'$\bf{{{self.name}}}$ ({self.lon}$^o$E, {self.lat}$^o$N): {self.start}')

        ax=plt.subplot(231)
        plt.plot(self.era5.thl[0,:], self.era5.z, label='ERA5')
        self.mxl_thl = Mixed_layer_profile(fig, ax, p_thl, z_thl, self.mxl_profs)
        plot_soundings('theta')
        plt.xlabel(r'$\theta_\mathrm{l}$ (K)')
        plt.ylabel(r'$z$ (m)')
        plt.legend()
        plt.ylim(0, z_top)

        ax=plt.subplot(232)
        plt.plot(self.era5.qt[0,:]*1e3, self.era5.z)
        self.mxl_qt = Mixed_layer_profile(fig, ax, p_qt, z_qt, self.mxl_profs)
        plot_soundings('qt', scale_fac=1000)
        plt.xlabel(r'$q_\mathrm{t}$ (g kg$^{-1}$)')
        plt.ylim(0, z_top)

        ax=plt.subplot(233)
        plt.plot(self.era5.time, self.era5.wth*1.2*1005, color='r', label=r'H')
        plt.plot(self.era5.time, self.era5.wq*1.2*2.5e6, color='b', label=r'LE')
        plt.xlabel(r'time (UTC)')
        plt.ylabel(r'Flux (W m$^{-2}$)')
        plt.legend()
        fmt_time()

        ax=plt.subplot(234)
        plt.plot(self.era5.u[0,:], self.era5.z, label=r'$u$')

        min_ug, mean_ug, max_ug = get_range(self.era5.ug)
        plt.fill_betweenx(self.era5.z, min_ug, max_ug, alpha=0.1)
        plt.plot(mean_ug, self.era5.z, ':', color='C0', label=r'$u_\mathrm{g}$')
        plot_soundings('u')

        self.mxl_u = Mixed_layer_profile(fig, ax, p_u, z_u, self.mxl_profs)
        plt.xlabel(r'$u$ (m s$^{-1}$)')
        plt.ylabel(r'$z$ (m)')
        plt.legend()
        plt.ylim(0, z_top)

        ax=plt.subplot(235)
        plt.plot(self.era5.v[0,:], self.era5.z, label=r'$v$')

        min_vg, mean_vg, max_vg = get_range(self.era5.vg)
        plt.fill_betweenx(self.era5.z, min_vg, max_vg, alpha=0.1)
        plt.plot(mean_vg, self.era5.z, ':', color='C0', label=r'$v_\mathrm{g}$')
        plot_soundings('v')

        self.mxl_v = Mixed_layer_profile(fig, ax, p_v, z_v, self.mxl_profs)
        plt.xlabel(r'$v$ (m s$^{-1}$)')
        plt.legend()
        plt.ylim(0, z_top)

        ax=plt.subplot(236)

        min_wls, mean_wls, max_wls = get_range(self.era5.wls)
        plt.fill_betweenx(self.era5.z, min_wls, max_wls, alpha=0.1)
        plt.plot(mean_wls, self.era5.z, ':', color='C0')

        self.mxl_wls = Subsidence_profile(fig, ax, self.mxl_ref['divU'])

        plt.xlabel(r'$w_\mathrm{LS}$ (m s$^{-1}$)')
        plt.ylim(0, z_top)

        # Keep track of all mixed-layer profiles. We need to 
        # update the boundary layer height in all plots.
        self.mxl_profs.append(self.mxl_thl) 
        self.mxl_profs.append(self.mxl_qt) 
        self.mxl_profs.append(self.mxl_u) 
        self.mxl_profs.append(self.mxl_v) 


    def save_to_json(self, z_top=4000):
        """
        Save mixed-layer parameters and other data (=sounding) to JSON after tuning..
        """ 

        # Get profiles from plotting instance.
        thl = self.mxl_thl.x
        z_thl = self.mxl_thl.y

        qt  = self.mxl_qt.x / 1000.
        z_qt  = self.mxl_qt.y

        u = self.mxl_u.x
        z_u = self.mxl_u.y

        v = self.mxl_v.x
        z_v = self.mxl_v.y

        # Store parameters.
        self.mxl_ref['h'] = float(z_thl[1])

        self.mxl_ref['theta'] = float(thl[0])
        self.mxl_ref['dtheta'] = float(thl[2] - thl[1])
        self.mxl_ref['gamma_theta'][0] = float((thl[3] - thl[2]) / (z_thl[3] - z_thl[2]))
        self.mxl_ref['gamma_theta'][1] = float((thl[4] - thl[3]) / (z_thl[4] - z_thl[3]))
        self.mxl_ref['z_theta'][0] = float(z_thl[3])
        self.mxl_ref['z_theta'][1] = z_top

        self.mxl_ref['qt'] = float(qt[0])
        self.mxl_ref['dqt'] = float(qt[2] - qt[1])
        self.mxl_ref['gamma_qt'][0] = float((qt[3] - qt[2]) / (z_qt[3] - z_qt[2]))
        self.mxl_ref['gamma_qt'][1] = float((qt[4] - qt[3]) / (z_qt[4] - z_qt[3]))
        self.mxl_ref['z_qt'][0] = float(z_qt[3])
        self.mxl_ref['z_qt'][1] = z_top

        self.mxl_ref['u'] = float(u[0])
        self.mxl_ref['du'] = float(u[2] - u[1])
        self.mxl_ref['gamma_u'][0] = float((u[3] - u[2]) / (z_u[3] - z_u[2]))
        self.mxl_ref['gamma_u'][1] = float((u[4] - u[3]) / (z_u[4] - z_u[3]))
        self.mxl_ref['z_u'][0] = float(z_u[3])
        self.mxl_ref['z_u'][1] = z_top
        self.mxl_ref['ug'] = self.mxl_ref['u'] + self.mxl_ref['du']

        self.mxl_ref['v'] = float(v[0])
        self.mxl_ref['dv'] = float(v[2] - v[1])
        self.mxl_ref['gamma_v'][0] = float((v[3] - v[2]) / (z_v[3] - z_v[2]))
        self.mxl_ref['gamma_v'][1] = float((v[4] - v[3]) / (z_v[4] - z_v[3]))
        self.mxl_ref['z_v'][0] = float(z_v[3])
        self.mxl_ref['z_v'][1] = z_top
        self.mxl_ref['vg'] = self.mxl_ref['v'] + self.mxl_ref['dv']

        self.mxl_ref['divU'] = float(-self.mxl_wls.x[1] / self.mxl_wls.y[1])

        self.mxl_ref['wtheta'] = [float(x) for x in self.era5.wth]
        self.mxl_ref['wq'] = [float(x) for x in self.era5.wq]

        self.mxl_ref['fc'] = float(self.era5.attrs['fc'])
        self.mxl_ref['p0'] = float(self.era5.ps.mean())

        self.mxl_ref['z0m'] = float(self.era5.z0m.mean())
        self.mxl_ref['z0h'] = float(self.era5.z0h.mean())

        self.mxl_ref['is_tuned'] = True

        # Add sounding (if available).
        if len(self.soundings) > 0:
            self.mxl['observations'] = []

            for name, df in self.soundings.items():

                # Interpolate sounding to reasonable height interval.
                z_out = np.arange(0, df['agl'].max(), 20)
                df_z = interpolate_sounding_to_height(df, z_out)

                d = dict(
                    name = name,
                    height = df_z.index.tolist(),
                    pressure = (df_z['pressure']/100).tolist(),
                    temperature = (df_z['temperature']-T0).tolist(),
                    relativeHumidity = df_z['relative_humidity'].tolist(),
                    windSpeed = df_z['speed'].tolist(),
                    windDirection = df_z['heading'].tolist()
                )

                self.mxl['observations'].append(d)

        # Save in JSON format.
        save_json(self.mxl, self.json_file)