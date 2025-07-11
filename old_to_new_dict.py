import glob
from help_code import load_json, save_json, get_web_dict


json_files = glob.glob('json/*.json')

for json_file in json_files:
    name = json_file.split('/')[-1].split('.')[0]
    old_json = load_json(json_file)

    new_json = get_web_dict()
    ref_dict = new_json['reference']

    for key in ref_dict.keys():
        ref_dict[key] = None

    ref_dict['name'] = name
    ref_dict['description'] = f'{name} {old_json["start_time"]} UTC.'

    ref_dict['h'] = old_json['zi']

    ref_dict['theta'] = old_json['thl']
    ref_dict['dtheta'] = old_json['dthl']
    ref_dict['gamma_theta'] = [old_json['gamma_thl_1'], old_json['gamma_thl_2']]
    ref_dict['z_theta'] = [old_json['z1_thl'], 3720],
    ref_dict['advtheta'] = 0.
    ref_dict['wtheta'] = old_json['wthl']

    ref_dict['qt'] = old_json['qt']
    ref_dict['dqt'] = old_json['dqt']
    ref_dict['gamma_qt'] = [old_json['gamma_qt_1'], old_json['gamma_qt_2']]
    ref_dict['z_qt'] = [old_json['z1_qt'], 3720],
    ref_dict['advq'] = 0.
    ref_dict['wq'] = old_json['wqt']

    ref_dict['u'] = old_json['u']
    ref_dict['ug'] = old_json['ug']
    ref_dict['du'] = old_json['du']
    ref_dict['gamma_u'] = [old_json['gamma_u_1'], old_json['gamma_u_2']]
    ref_dict['z_u'] = [old_json['z1_u'], 3720],
    ref_dict['advu'] = 0.

    ref_dict['v'] = old_json['v']
    ref_dict['vg'] = old_json['vg']
    ref_dict['dv'] = old_json['dv']
    ref_dict['gamma_v'] = [old_json['gamma_v_1'], old_json['gamma_v_2']]
    ref_dict['z_v'] = [old_json['z1_v'], 3720],
    ref_dict['advv'] = 0.

    ref_dict['z0m'] = old_json['z0m']
    ref_dict['z0h'] = old_json['z0h']
    ref_dict['ustar'] = 0.1

    ref_dict['divU'] = old_json['div']
    ref_dict['fc'] = old_json['fc']
    ref_dict['p0'] = old_json['ps']

    ref_dict['runtime'] = old_json['time'][-1] * 3600

    ref_dict['is_tuned'] = True

    for key,value in ref_dict.items():
        if value is None:
            print('Warning: Key "{}" has no value.'.format(key))

    save_json(new_json, f'json/{name}_web.json')