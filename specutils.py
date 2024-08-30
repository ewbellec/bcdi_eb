import numpy as np
import pylab as plt
from silx.io.specfile import SpecFile as SF
import fabio
from silx.io.spech5 import SpecH5


# def openSpec(specfile, printing=True):
def get_scans_title_str_hdf5(specfile, printing=True):
    spec = SF(specfile)
    
    if printing==True:
        for scan_no in spec.keys():
            for line in spec[scan_no].header:
                if '#S' in line:
                    print(line)
    return spec

def get_command_spec(scan_no, specfile):
    """
    return the scan command/title
    """
    spec = SF(specfile)
    for line in spec[scan_no-1].header:
        if '#S' in line:
            break
    command = ''.join([l + ' ' for l in line.split()[2:]])
    return command[:-1]

class Scan_spec:
    def __init__(self, specfile, scan_nb,
                 path_imgs=None, search_detector=True,
                 verbose=True):
        """
        filename = path to hdf5 file (dataset level)
        scan_nb = scan number (accepts -n nomenclature)
        verbose = show some output
        """
        self.specfile = specfile
        self.h5file = self.specfile[:-5] + '.h5' # Dummy h5file for saving function
        self.spec = SF(self.specfile)
        self.scan_string = "{}.1".format(scan_nb)
        self.path_imgs = path_imgs
        self.verbose = verbose
        
        self.getCommand()
        self.getSampleName()
        
        self.getFirstImageFile()
        self.detector = self.img_first_file.split('_')[-2]
            

        if verbose:
            print(f"Scan no: {self.scan_string}")
            print(f"command: {self.command}")
            
            
    def getCommand(self):
        for line in self.spec[self.scan_string].scan_header:
            if '#S' in line:
                self.command = line
        return self.command
    
    def getSampleName(self):
        for line in self.spec[self.scan_string].header:
            if '#F' in line:
                break
        sample = line.split('/')[-4]
        self.sample = sample
        return sample
    
    def getEnergy(self):
        for line in self.spec[self.scan_string].scan_header:
            if '#UMONO' in line:
                break
        self.energy = float(line.split('mononrj=')[-1].split('keV')[0]) * 1e3 # energy in eV
        return self.energy
    
    def getMotorPosition(self, motor_name):
        try:
            motor_pos = SpecH5(self.specfile)[self.scan_string]['measurement'][motor_name][()]
        except:
            motor_pos = self.spec[self.scan_string].motor_position_by_name(motor_name)
        return motor_pos
        
    #######################################################################################################################################
    ##########################################          Detector images functions          ################################################
    #######################################################################################################################################

    def getFirstImageFile(self):
        for line in self.spec[self.scan_string].scan_header:
            if 'ULIMA' in line:
                img_first_file = line.split()[-1]
        self.img_first_file = img_first_file
        return
        
    def getImagesNumbers(self):
        counter_dict = {'eiger2M' : 'ei2minr',
                        'mpx4' : 'mpx4inr',
                        'mpx22' : 'mpx22ir'}

        counter = counter_dict[self.detector]

        img_nb_list = self.spec[self.scan_string].data_column_by_name(counter)
        return img_nb_list

    def getImagesFilesList(self):
#         self.getFirstImageFile()
        
        prefix = self.img_first_file.split('.')[0][:-5]
        if self.path_imgs is not None:
            prefix = self.path_imgs + prefix.split('/')[-1]
        
        if '.edf.gz' in self.img_first_file :
            suffix = '.edf.gz'
        elif '.edf' in self.img_first_file:
            suffix = '.edf'
        elif '.h5' in self.img_first_file:
            suffix = '.h5'
        elif '.h5bs' in self.img_first_file:
            suffix = '.h5bs'
        else:
            suffix = ''

        img_path_list = []
        img_nb_list = self.getImagesNumbers()

        # Test if there's an error
        first_nb = int(self.img_first_file.split('.')[0].split('_')[-1])
        if first_nb != img_nb_list[0]:
            print('Error : first image number from GetImageNumbers seems wrong !')

        for img_nb in img_nb_list:
            img_path_list.append(prefix+"%05d" % img_nb + suffix)
        self.img_path_list = img_path_list
        self.suffix = suffix
        return 


    def getImagesRaw(self):

        self.getImagesFilesList() 

        for n,img_path in enumerate(self.img_path_list):
            if self.suffix=='.edf' or self.suffix=='.edf.gz' or self.suffix=='' or self.suffix=='.h5bs':
                img = fabio.open(img_path).data
            elif suffix=='.h5':
                hdf5 = h5py.File(img_path)
                img = hdf5[list(hdf5.keys())[0]]['measurement']['data'].__array__()[0]

            if n==0:
                data = np.zeros((len(self.img_path_list),)+img.shape)
            data[n] += img
            
        if self.verbose:
            print("data.shape", data.shape)
            
        return data
            
    def getImages(self):

        data = self.getImagesRaw()

        if self.verbose:
            print("data.shape", data.shape)
        return data

    #######################################################################################################################################
    ##########################################                    ################################################
    #######################################################################################################################################

    def getDetCalibInfo(self):
        det_calib = {}
        for line in self.spec[self.scan_string].scan_header:
            if '#UDETCALIB' in line:
                break

        det_calib['beam_center_x'] = float(line.split()[1].split(',')[0].split('=')[1])
        det_calib['beam_center_y'] = float(line.split()[1].split(',')[1].split('=')[1])
#         pixperdeg = float(line.split()[1].split(',')[2].split('=')[1])
        det_distance_CC = float(line.split()[1].split(',')[3].split('=')[1])
        det_distance_COM = float(line.split()[1].split(',')[4].split('=')[1])
        det_calib['distance'] = det_distance_COM
#         timestamp = line.split()[1].split(',')[5].split('=')[1]

        if self.detector == 'mpx4' or self.detector == 'mpx22':
            pixel_sizes = 55e-6
        elif self.detector == 'eiger2M':
            pixel_sizes = 75e-6
        else:
            raise ValueError('detector\'s pixel sizes unknown')
        det_calib['x_pixel_size'] = pixel_sizes
        det_calib['y_pixel_size'] = pixel_sizes
        
        self.det_calib = det_calib
        if self.verbose:
            print(self.det_calib)
        return det_calib
    
    
#######################################################################################################################################
##########################################          Standard Scan class          ################################################
#######################################################################################################################################

class StandardScan_spec(Scan_spec):
    def __init__(self, filename, scan_nb, path_imgs=None, verbose=False):
        super().__init__(filename, scan_nb, path_imgs=path_imgs, verbose=verbose)

        _ = self.getDscanMotorPosition()

    def getDscanMotorPosition(self):
        motor_name = self.command.split()[3]
        motor = self.spec[self.scan_string].data_column_by_name(motor_name)

        if self.verbose:
            print("motor : {}".format(motor_name))

        self.motor = motor
        self.motor_name = motor_name

        return motor, motor_name

#     def getRoiData(self, roi_name, plot=False, fig_title=''):
#         with h5.File(self.h5file, "r") as h5f:
#             roidata = h5f[self.scan_string]["measurement/{}".format(roi_name)][()]
#         if plot:
#             plt.figure()
#             plt.plot(self.motor, roidata, ".-")
#             plt.xlabel(self.motor_name, fontsize=15)
#             plt.ylabel(roi_name, fontsize=15)
#             plt.title(fig_title, fontsize=15)
#         return roidata


##################################################################################################################################
#####################################           Dmesh scan          #######################################################
##################################################################################################################################

class DmeshScan_spec(Scan_spec):
    def __init__(self, filename, scan_nb, path_imgs=None, verbose=True):
        super().__init__(filename, scan_nb, path_imgs=path_imgs, verbose=verbose)

        _ = self.getMeshMotorPosition()

    def getMeshMotorPosition(self):
        motor1_name = self.command.split()[3]
        motor2_name = self.command.split()[7]

        motor1 = self.spec[self.scan_string].data_column_by_name(motor1_name)
        motor2 = self.spec[self.scan_string].data_column_by_name(motor2_name)

        shape = (int(self.command.split()[-2]) + 1, int(self.command.split()[6]) + 1)
        motor1 = np.reshape(motor1, shape)
        motor2 = np.reshape(motor2, shape)

        if self.verbose:
            print("motor1 : {}".format(motor1_name))
            print("motor2 : {}".format(motor2_name))

        self.motor1 = motor1
        self.motor2 = motor2
        self.motor1_name = motor1_name
        self.motor2_name = motor2_name

        return motor1, motor2, motor1_name, motor2_name

    def getImages(self):
        data = self.getImagesRaw()
        data = np.reshape(data, (self.motor1.shape) + data.shape[-2:])

        if self.verbose:
            print("data.shape", data.shape)
            print(
                "shape dimensions : ( {}, {}, {}, {})".format(
                    self.motor2_name,
                    self.motor1_name,
                    "detector vertical axis",
                    "detector horizontal axis",
                )
            )
        return data
    
    
##################################################################################################################################
#################################           General open scan function          ##################################################
##################################################################################################################################
    
    
def openScan(filename, scan_nb, verbose=False):
    command = get_command_spec(scan_nb, filename)

    if "scan" in command and "lookupscan" not in command and "loopscan" not in command:
        return StandardScan_spec(filename, scan_nb, verbose=verbose)

#     if "lookupscan" in command:
#         return LookupScan(filename, scan_nb, verbose=verbose)

    if "mesh" in command:
        return DmeshScan_spec(filename, scan_nb, verbose=verbose)

#     if "sxdm" in command or "kmap" in command:
#         return SXDM_Scan(filename, scan_nb, verbose=verbose)

#     if "ct" in command:
#         return Scan_ct(filename, scan_nb, verbose=verbose)
    
#     if "loopscan" in command:
#         return Scan(filename, scan_nb, verbose=verbose)