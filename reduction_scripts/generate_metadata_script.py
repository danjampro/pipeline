"""
Example:
python generate_metadata_script_marusa.py config.py /priv/mulga1/marusa/2m3data/20190114
"""
import sys
import getopt
import os
import imp

import numpy as np
from astropy.io import fits as pyfits

from pywifes import wifes_calib


STD_STAR_MAP = {"HD31128": "HD031128"}


def _parse_obj_name(object_name):
    """
    """
    try:
        return STD_STAR_MAP[object_name]
    except KeyError:
        return object_name

# List of standard stars
stdstar_list = wifes_calib.ref_fname_lookup.keys()
stdstar_is_flux_cal = wifes_calib.ref_flux_lookup
stdstar_is_telluric = wifes_calib.ref_telluric_lookup

# Config file
config = imp.load_source('config', sys.argv[1])
metadata = config.generate_metadata

# In case calibration files are missing, specify dates (format 20190315) for each cal type
selected_cal_dates = {}
try:
    # ~ opts, args = getopt.getopt(sys.argv,"dark:bias:flat:",["dark=", "bias=", "flat="])
    opts, args = getopt.getopt(sys.argv[3:], "d:b:f:")
except getopt.GetoptError:
    print('test.py -i <inputfile> -o <outputfile>')
for opt, arg in opts:
    if opt in ("-d"):
        selected_cal_dates['DARK'] = int(arg)
    elif opt in ("-b"):
        selected_cal_dates['BIAS'] = int(arg)
    elif opt in ("-f"):
        selected_cal_dates['FLAT'] = int(arg)

print('SELECTED_CAL_DATES', selected_cal_dates)


ccdsum = metadata['CCDSUM']  # '1 1' # '1 2' # binning
prefix = metadata['prefix']

# If you wish to reduce only selected objects
objectnames = metadata['objectnames']
if objectnames:
    objectnames = [x.replace(' ', '') for x in objectnames]

# If you wish to exclude selected objects
exclude_objectnames = metadata['exclude_objectnames']
if exclude_objectnames:
    exclude_objectnames = [x.replace(' ', '') for x in exclude_objectnames]


# Input folder with raw data
data_dir = os.path.join(config.input_root, config.OBSDATE)

print('#'+54*'-')
print('data_dir', data_dir)

# Output folder (should already exist and metadata should be there)
root_obsdate = os.path.join(config.output_root, str(config.OBSDATE))

# Create folder with date
if not os.path.isdir(root_obsdate):
    os.mkdir(root_obsdate)
print('root_obsdate', root_obsdate)

# Add band (grating)
out_dir = os.path.join(root_obsdate, 'reduced_%s' % config.band)

if prefix is not None and len(prefix) > 0:
    print('prefix', prefix)
    out_dir += '_%s' % prefix
out_dir_bool = os.path.isdir(out_dir) and os.path.exists(out_dir)
if not out_dir_bool:
    os.mkdir(out_dir)
print('out_dir', out_dir)

######################################

"""
Often you don't take all calib frames in the same night.
Make a folder just with all calib frames from the run. Then select ones with the closest data
(preferrably the same date as science frames).
"""


print('#'+54*'-')
print('data_dir: %s' % data_dir)
print('prefix: %s' % metadata['prefix'])
print('objectnames:', objectnames)
print('#'+54*'-')

# get list of all fits files in directory
all_files = [os.path.join(data_dir, x) for x in os.listdir(data_dir)]


def find_all_modes():
    """
    Find all different modes taken during this night.
    By modes I mean different combinations of gratings, binning etc.
    Find images that go together.

    NOTE that some images were taken with echelle. --> Actually not. ECHELLE DATA ARE MISSING!?!
    """
    modes = dict()

    keywords = ['NAXIS1', 'NAXIS2', 'WINDOW', 'GRATINGB', 'GRATINGR', 'BEAMSPLT', 'CCDSIZE',
                'CCDSEC', 'CCDSUM', 'TRIMSEC', 'DATASEC', 'DETSEC']
    # keywords=['IMAGETYP', 'NAXIS1', 'NAXIS2', 'WINDOW', 'GRATINGB', 'GRATINGR', 'CCDSEC',
    # 'CCDSUM', 'TRIMSEC', 'DATASEC', 'DETSEC']

    for fn in all_files:
        try:
            f = pyfits.open(fn)
            header = f[0].header
            f.close()
        except Exception:
            continue

        try:
            k = [header[x] for x in keywords]
        except Exception:
            print('Cannot get full header for', fn, header['IMAGETYP'])
            continue

        k = tuple(k)  # Lists or sets cannot be dictionary keys
        try:
            modes[k].append(fn)
        except Exception:
            modes[k] = [fn]

    return modes


def find_filenames():
    blue_obs = []
    red_obs = []
    obs_date = None
    for fn in all_files:
        obs = fn.replace('.fits', '').split('/')[-1]
        if obs_date == None:
            try:
                #~ f = pyfits.open(data_dir+fn)
                f = pyfits.open(fn)
                obs_date = f[0].header['DATE-OBS'].split('T')[0].replace('-', '')
                f.close()
            except:
                continue
            #obs_date = obs[7:15]
        try:
            f = pyfits.open(fn)
            camera = f[0].header['CAMERA']
            f.close()
        except:
            continue
        if camera == 'WiFeSBlue':
            if obs in blue_obs:
                continue
            else:
                blue_obs.append(obs)
        if camera == 'WiFeSRed':
            if obs in red_obs:
                continue
            else:
                red_obs.append(obs)


    return blue_obs, red_obs, obs_date


def find_filenames_for_a_mode(all_files):
    """
    """
    blue_obs = []
    red_obs = []
    obs_date = None
    for fn in all_files:
        obs = fn.replace('.fits', '').split('/')[-1]
        if obs_date == None:
            try:
                #~ f = pyfits.open(data_dir+fn)
                f = pyfits.open(fn)
                obs_date = f[0].header['DATE-OBS'].split('T')[0].replace('-', '')
                f.close()
            except:
                continue
            #obs_date = obs[7:15]
        try:
            f = pyfits.open(fn)
            camera = f[0].header['CAMERA']
            f.close()
        except:
            continue
        if camera == 'WiFeSBlue':
            if obs in blue_obs:
                continue
            else:
                blue_obs.append(obs)
        if camera == 'WiFeSRed':
            if obs in red_obs:
                continue
            else:
                red_obs.append(obs)


    return blue_obs, red_obs, obs_date



# Marusa: Match science images and arcs. One arc before and one after the science exposure + all arcs in between.
# Disadvantage: if you e.g. observe 1 objects with arcs, then do other things and observe it later again, everything is combined together.
def match_object_and_arc(objects=None, arcs=None):
    """
    Find one arc before and one arc after science exposures + all arxs in between. Based on MJD comparison.
    """
    result={}

    arc_mjd=np.array([x[1] for x in arcs])

    for k, v in objects.items():
        v=sorted(v)

        mjd_first_science = v[0][1]
        mjd_last_science = v[-1][1]
        exptime_last=v[-1][2]/3600.0/24.0 # convert exptime to days
        mjd_last_science+=exptime_last

        a=[]

        # Find arc taken just before the first science exposure
        diff_first=mjd_first_science-arc_mjd # Take first positive
        mask_first = diff_first>0
        if len(diff_first[mask_first])>0:
            value_first=sorted(diff_first[mask_first])[0]
            index_first=np.where(diff_first==value_first)[0][0]
            a=[arcs[index_first][-1]]
        else:
            pass # First image of the night. First arc was taken later.

        # Find arc taken just after the last science exposure.
        diff_last=arc_mjd-mjd_last_science # Take closest to 0 from the negative side
        mask_last = diff_last>0
        if len(diff_last[mask_last])>0:
            value_last=sorted(diff_last[mask_last])[0]
            index_last=np.where(diff_last==value_last)[0][0]
            a.append(arcs[index_last][-1])
        else:
            pass # No arc at the end of the night. This shouldn't happen.

        # Check if there were any other arcs between first and last science exposure.
        mask = (arc_mjd>mjd_first_science) & (arc_mjd<mjd_last_science)
        for m, ar in zip(mask, arcs):
            if m:
                a.append(ar[-1])

        result[k]=a
    return result

def test_if_all_essential_calib_files_are_available(camera=None, science=None, arcs=None, dark=None, bias=None, flat_dome=None, flat_twi=None, std_obs=None, wire=None):
    result={}

    if len(science)<1:
        print('**** WARNING (%s): No science frames found.'%camera)

    if len(arcs)<1:
        print('**** WARNING (%s): No arc frames found.'%camera)

    if len(bias)<1:
        print('**** WARNING (%s): No bias frames found.'%camera)
        bias_key=False
        result['BIAS']=False
    else:
        bias_key=True
        result['BIAS']=bias

    if len(dark)<1:
        print('**** WARNING (%s): No dark frames found.'%camera)
        dark_key=False
        result['DARK']=False
    else:
        dark_key=True
        result['DARK']=dark

    if len(flat_dome)<1:
        print('**** WARNING (%s): No dome flat frames found.'%camera)
        flat_dome_key=False
        result['FLAT']=False
    else:
        flat_dome_key=True
        result['FLAT']=flat_dome

    if len(flat_twi)<1:
        print('WARNING (%s): No twilight flat frames found.'%camera)

    if len(wire)<1:
        print('WARNING (%s): No wire frames found.'%camera)

    if len(std_obs)<1:
        print('WARNING (%s): No std_obs frames found.'%camera)

    return result


def classify_frames_into_imagetypes(frames):
    """
    """
    bias = []
    domeflat = []
    twiflat = []
    dark = []
    arc = []
    wire = []
    stdstar = {}
    science = {}
    objects = {}
    arcs = []

    for obs in frames:

        filename = os.path.join(data_dir, obs + '.fits')
        header = pyfits.getheader(filename)

        obj_name = _parse_obj_name(header['OBJECT'].upper())
        try:
            mjd = header['MJD-OBS']
            run = header['RUN']
            imagetype = header['IMAGETYP'].upper()
            exptime = header['EXPTIME']
        except KeyError as err:
            print(f"Unable to parse header for {obj_name}: {err}")
        # naxis2 = header['NAXIS2']
        # ccdsumf = header['CCDSUM']  # binning: '1 1' or '1 2'

        if imagetype == 'OBJECT':
            if objectnames:  # keep only selected objects
                if obj_name not in objectnames:
                    continue
            if exclude_objectnames:  # exclude selected objects
                if obj_name in exclude_objectnames:
                    continue

        # 1 - bias frames
        if imagetype == 'ZERO':
            print(f"Identified object as bias: {obj_name}.")
            bias.append(obs)

        # 2 - quartz flats
        if imagetype == 'FLAT':
            print(f"Identified object as flat: {obj_name}.")
            domeflat.append(obs)

        # 3 - twilight flats
        if imagetype == 'SKYFLAT':
            print(f"Identified object as sky flat: {obj_name}.")
            twiflat.append(obs)

        # 4 - dark frames
        if imagetype == 'DARK':
            print(f"Identified object as dark: {obj_name}.")
            dark.append(obs)

        # 5 - arc frames
        if imagetype == 'ARC':
            print(f"Identified object as arc: {obj_name}.")
            arc.append(obs)
            arcs.append([run, mjd, imagetype, obs])

        # 6 - wire frames
        if imagetype == 'WIRE':
            print(f"Identified object as wire: {obj_name}.")
            wire.append(obs)

        # all else are science targets
        if imagetype == 'OBJECT':

            if obj_name in stdstar_list:
                print(f"Identified standard star for object: {obj_name}.")
                # group standard obs together!
                if obj_name in stdstar.keys():
                    stdstar[obj_name].append(obs)
                else:
                    stdstar[obj_name] = [obs]
            else:
                print(f"Identified science object: {obj_name}.")
                # group science obs together!
                if obj_name in science.keys():
                    science[obj_name].append(obs)
                else:
                    science[obj_name] = [obs]

            # For arc-science matching
            if obj_name in objects.keys():
                objects[obj_name].append([run, mjd, exptime, imagetype, obs])
            else:
                objects[obj_name] = [[run, mjd, exptime, imagetype, obs]]

    arcs_per_star = match_object_and_arc(objects=objects, arcs=arcs)

    return science, bias, domeflat, twiflat, dark, arc, wire, stdstar, arcs_per_star


def write_metadata(science=None, bias=None, domeflat=None, twiflat=None, dark=None, arc=None, arcs_per_star=None, wire=None, camera=None, std_obs=None, nmode=0, kmode=None):
    if len(science)>0:
        pass
    else:
        print('No science images for the mode', kmode)
        return False
    #------------------------------------------------------
    # write to metadata save script!
    #~ f = open('save_blue_metadata.py', 'w')
    if prefix is not None and len(prefix)>0:
        metadata_filename=os.path.join(out_dir, '%s_mode_%d_metadata_%s.py'%(prefix, nmode, camera))
    else:
        metadata_filename=os.path.join(out_dir, 'mode_%d_metadata_%s.py'%(nmode, camera))
    #~ f = open('%s_%s.py'%(config.metadata_filename, camera), 'w')
    f = open(metadata_filename, 'w')

    dsplit = '#' + 54*'-' + '\n'

    #------------------
    # mode
    f.write(dsplit)
    f.write('mode = '+str(kmode)+'\n\n')

    #------------------
    # calibrations
    f.write(dsplit)

    # 1 - bias
    f.write('bias_obs = [\n')
    for obs in bias:
        f.write('   \'%s\', \n' % obs)
    f.write('    ]\n')
    f.write('\n')

    # 2 - domeflat
    f.write('domeflat_obs = [\n')
    for obs in domeflat:
        f.write('   \'%s\', \n' % obs)
    f.write('    ]\n')
    f.write('\n')

    # 3 - twiflat
    f.write('twiflat_obs = [\n')
    for obs in twiflat:
        f.write('   \'%s\', \n' % obs)
    f.write('    ]\n')
    f.write('\n')

    # 4 - dark
    f.write('dark_obs = [\n')
    for obs in dark:
        f.write('   \'%s\', \n' % obs)
    f.write('    ]\n')
    f.write('\n')

    # 5 - arc
    f.write('arc_obs = [\n')
    for obs in arc:
        f.write('   \'%s\', \n' % obs)
    f.write('    ]\n')
    f.write('\n')

    # 6 - wire
    f.write('wire_obs = [\n')
    for obs in wire:
        f.write('   \'%s\', \n' % obs)
    f.write('    ]\n')
    f.write('\n')

    #------------------
    # science
    #~ f.write(dsplit)
    f.write('sci_obs = [\n')
    for obj_name in science.keys():
        obs_list = science[obj_name]
        obs_str = '\'%s\'' % obs_list[0]
        for i in range(1,len(obs_list)):
            obs = obs_list[i]
            obs_str += ',\n               \'%s\'' % obs
        obs_list_arc = arcs_per_star[obj_name]
        obs_arc_str = '\'%s\'' % obs_list_arc[0]
        for i in range(1,len(obs_list_arc)):
            obs = obs_list_arc[i]
            obs_arc_str += ',\n               \'%s\'' % obs
        f.write('    # %s\n' % obj_name)
        f.write('    {\'sci\'  : [%s],\n' % obs_str)
        f.write('     \'sky\'  : [],\n')
        f.write('     \'arc\'  : [%s]},\n' % obs_arc_str)


    f.write('    ]')
    f.write('\n\n')

    #------------------
    # stdstars
    #~ f.write(dsplit)
    f.write('std_obs = [')
    for obj_name in stdstar.keys():
        obs_list = stdstar[obj_name]
        obs_str = '\'%s\'' % obs_list[0]
        for i in range(1,len(obs_list)):
            obs = obs_list[i]
            obs_str += ',\n               \'%s\'' % obs
         # Also write arcs for tellurics
        obs_list_arc = arcs_per_star[obj_name]
        obs_arc_str = '\'%s\'' % obs_list_arc[0]
        for i in range(1,len(obs_list_arc)):
            obs = obs_list_arc[i]
            obs_arc_str += ',\n               \'%s\'' % obs

        f.write('    # %s\n' % obj_name)
        f.write('    {\'sci\'  : [%s],\n' % obs_str)
        f.write('     \'name\' : [\'%s\'],\n' % obj_name)
        # Star is both telluric and flux standard
        if stdstar_is_flux_cal[obj_name] and stdstar_is_telluric[obj_name]:
            f.write('     \'type\' : [\'flux\', \'telluric\'],\n')
        # Star is flux standard but not telluric
        elif stdstar_is_flux_cal[obj_name] and not stdstar_is_telluric[obj_name]:
            f.write('     \'type\' : [\'flux\'],\n')
        # Star is telluric but not flux standard
        elif not stdstar_is_flux_cal[obj_name] and stdstar_is_telluric[obj_name]:
            f.write('     \'type\' : [\'telluric\'],\n')
        f.write('     \'arc\'  : [%s]},\n' % obs_arc_str)
    f.write('    ]\n')
    f.write('\n')

    #------------------
    # footers
    f.write(dsplit)
    #~ out_fn = 'wifesB_%s_metadata.pkl' % obs_date
    f.write('night_data = {\n')
    f.write('    \'bias\' : bias_obs,\n')
    f.write('    \'domeflat\' : domeflat_obs,\n')
    f.write('    \'twiflat\' : twiflat_obs,\n')
    f.write('    \'dark\' : dark_obs,\n')
    f.write('    \'wire\' : wire_obs,\n')
    f.write('    \'arc\'  : arc_obs,\n')
    f.write('    \'sci\'  : sci_obs,\n')
    f.write('    \'std\'  : std_obs}\n')
    f.write('\n')
    #~ f.write('f1 = open(\'%s\', \'w\')' % out_fn
    #~ f.write('pickle.dump(night_data, f1)'
    #~ f.write('f1.close()'

    f.close()

    print('METADATA written in', metadata_filename)
    return True


def propose_missing_calib_files(mode=None, calstat=None):
    """
    calstat: stats on calibrations: calstat[imagetype]=False/len(images)
    """
    print

    # Calibrations
    try:
        c=cal[mode]
    except:
        c=None

    # What calib files are missing?
    for imagetype, status in calstat.items():
        if status: # Set lower limit on number of calib files needed.
            if len(status)<3:
                missing=True
                TODO=True
        else: # Missing. Find them. What if c==None?
            try:
                dates=c[imagetype] # dates available for this particular imagetype
                print('Missing %s calibration file. Available:' % imagetype)
                for k, v in dates.items():
                    print(k, len(v)) # len is both blue and red!
                    #~ filenames=v[date]

                print
            except:
                if imagetype=='BIAS': # zero instead of bias
                    try:
                        dates=c['ZERO'] # dates available for this particular imagetype
                        print('Missing %s calibration file. Available:'%imagetype)
                        for k, v in dates.items():
                            print(k, len(v)) # len is both blue and red!
                            #~ filenames=v[date]

                        print
                    except:
                        print('*** NO %s AVAILABLE FOR THIS MODE!!!' % imagetype)

                else:
                    print('*** NO %s AVAILABLE FOR THIS MODE!!!' % imagetype)


def include_missing_calib_files(mode=None, calstat=None, camera=None):
    """
    calstat: stats on calibrations: calstat[imagetype]=False/len(images)
    """
    result = {}

    """
    # Calibrations
    try:
        c = cal[mode]
    except Exception:
        c = None
        print('ERROR: no calibration files found for this mode:', mode)
        return False
    """

    # What calib files are missing?
    for imagetype, status in calstat.items():
        if imagetype not in selected_cal_dates:
            continue
        else:
            if imagetype not in ['BIAS', 'ZERO']:
                date_wanted = selected_cal_dates[imagetype]  # check BIAS-ZERO
            else:
                try:
                    date_wanted = selected_cal_dates['ZERO']  # check BIAS-ZERO
                except Exception:
                    date_wanted = selected_cal_dates['BIAS']  # check BIAS-ZERO

        if status:
            pass

        else:  # Missing. Find them. What if c==None?
            try:
                dates = c[imagetype]  # dates available for this particular imagetype
                filenames = dates[date_wanted]

                # Take only a selected band (blue or red)
                if camera == 'WiFeSRed':
                    filenames = [x for x in filenames if 'T2m3wr' in x]
                elif camera == 'WiFeSBlue':
                    filenames = [x for x in filenames if 'T2m3wb' in x]

                # Delete path. It is added later in the reduction code
                filenames = [x.split('/')[-1].replace('.fits', '') for x in filenames]

                print('Adding %d %s images:' % (len(filenames), imagetype))
                result[imagetype] = filenames
                for x in filenames:
                    print(x)

            except Exception:
                if imagetype == 'BIAS':  # zero instead of bias
                    try:
                        dates = c['ZERO']  # dates available for this particular imagetype
                        filenames = dates[date_wanted]

                        # Take only a selected band (blue or red)
                        if camera == 'WiFeSRed':
                            filenames = [x for x in filenames if 'T2m3wr' in x]
                        elif camera == 'WiFeSBlue':
                            filenames = [x for x in filenames if 'T2m3wb' in x]

                        filenames = [x.split('/')[-1].replace('.fits', '') for x in filenames]

                        print('Adding %d %s images:' % (len(filenames), imagetype))
                        result[imagetype] = filenames
                        for x in filenames:
                            print(x)
                        print
                    except Exception:
                        print('*** NO %s AVAILABLE FOR THIS MODE!!!' % imagetype)

                else:
                    print('*** NO %s AVAILABLE FOR THIS MODE!!!' % imagetype)

    return result


def find_missing_calib_files(mode=None, selected_cal_dates=None):
    """
    """
    # Calibrations
    try:
        c = cal[mode]
    except Exception:
        c = None

    result = {}
    if c:
        for imagetype, date in selected_cal_dates.items():  # For imagetype, dict of dates d
            tmp = c[imagetype]
            filenames = tmp[date]  # filenames for date
            result[imagetype] = filenames
    else:
        print('No calibration frames found.')

    return result


if __name__ == '__main__':

    modes = find_all_modes()

    m = 0
    for mode, v in modes.items():
        blue_obs, red_obs, obs_date = find_filenames_for_a_mode(v)

        if config.band == 'r':
            camera = 'WiFeSRed'
            science, bias, domeflat, twiflat, dark, arc, wire, stdstar, arcs_per_star = classify_frames_into_imagetypes(frames=red_obs)

        elif config.band == 'b':
            camera = 'WiFeSBlue'
            science, bias, domeflat, twiflat, dark, arc, wire, stdstar, arcs_per_star = classify_frames_into_imagetypes(frames=blue_obs)

        if len(science) > 0:
            print('########################################################')
            print('Mode', mode)
            print('########################################################')

            print('#'+54*'-')
            print('OBSDATE', config.OBSDATE)
            print('%s: %d objects found:' % (camera, len(science)))
            for k in science.keys():
                print(k)
            print('#'+54*'-')
        else:
            print(f"No science data found for mode: {mode}.")



    #~ blue_obs, red_obs, obs_date = find_filenames()

    # Blue camera
    #~ camera='WiFeSBlue'
    #~ science, bias, domeflat, twiflat, dark, arc, wire, stdstar, arcs_per_star = classify_frames_into_imagetypes(frames=blue_obs)
    #~ if len(science)>0:
        #~ print('%s: %d objects found:'%(camera, len(science)))
        #~ for k in science.keys():
            #~ print(k)
        #~ print('#'+54*'-')
    #~ bias_key, dark_key, flat_dome_key = test_if_all_essential_calib_files_are_available(camera=camera, science=science, arcs=arc, dark=dark, bias=bias, flat_dome=domeflat, flat_twi=twiflat, std_obs=stdstar, wire=wire)

    #~ if np.any(bias_key, dark_key, flat_dome_key):

    #~ write_metadata(camera=camera, science=science, bias=bias, domeflat=domeflat, twiflat=twiflat, dark=dark, arc=arc, arcs_per_star=arcs_per_star, wire=wire, std_obs=stdstar)

        # Red camera
        #~ camera='WiFeSRed'

        # Availability of calibration files
        calstat = test_if_all_essential_calib_files_are_available(
            camera=camera, science=science, arcs=arc, dark=dark, bias=bias, flat_dome=domeflat,
            flat_twi=twiflat, std_obs=stdstar, wire=wire)

        propose_missing_calib_files(mode=mode, calstat=calstat)

        # Update array with missing data
        missing_cal = include_missing_calib_files(mode=mode, calstat=calstat, camera=camera)
        print(missing_cal)
        for imagetype, filenames in missing_cal.items():
            if imagetype == 'BIAS':
                bias = filenames
                print('new biases'), bias
            elif imagetype == 'DARK':
                dark = filenames
                print('new darks'), dark
            elif imagetype == 'FLAT':
                domeflat = filenames
                print('new flats'), domeflat
            else:
                print("Update missing data: We've got a problem here.", imagetype)

        # SORT filenames
        bias = sorted(bias)
        domeflat = sorted(domeflat)
        dark = sorted(dark)
        arc = sorted(arc)

        success = write_metadata(camera=camera, science=science, bias=bias, domeflat=domeflat,
                                 twiflat=twiflat, dark=dark, arc=arc, arcs_per_star=arcs_per_star,
                                 wire=wire, std_obs=stdstar, nmode=m, kmode=mode)

        #~ if success:
            #~ test_if_all_essential_calib_files_are_available(camera=camera, science=science, arcs=arc, dark=dark, bias=bias, flat_dome=domeflat, flat_twi=twiflat, std_obs=stdstar, wire=wire)

        #~ success = write_metadata(camera=camera, science=science, bias=bias, domeflat=domeflat, twiflat=twiflat, dark=dark, arc=arc, arcs_per_star=arcs_per_star, wire=wire, std_obs=stdstar, nmode=m, kmode=k)

        #~ if success:
            #~ test_if_all_essential_calib_files_are_available(camera=camera, science=science, arcs=arc, dark=dark, bias=bias, flat_dome=domeflat, flat_twi=twiflat, std_obs=stdstar, wire=wire)

        m += 1 # mode numbers
